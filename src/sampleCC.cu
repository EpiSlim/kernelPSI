#define ARMA_ALLOW_FAKE_GCC

#include <chrono>
#include <ctime>
#include <ratio>
using namespace std::chrono;

// C/C++ headers
#include <cmath>
#include <float.h>

#include <RcppArmadillo.h>
//#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// CUDA headers
#include "cublas_v2.h"
#include "curand.h"
#include <cuda.h>
#include <cuda_runtime.h>

// #include "maxmin.hpp"

// CUDA kernels
__global__ void cuda_affine_trans(double *lambda, int n, double *a, double *b, 
                                  double *c) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    c[i] = a[i] + *lambda * b[i];
}

__global__ void cuda_inverse_cdf(int n, double *a, double *b, double mu,
                                 double sigma) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    a[i] = normcdfinv((b[i] - mu) / sigma);
}

__global__ void cuda_vector_normalize(int n, double *norm2, double *a) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    a[i] /= *norm2;
}

__global__ void cuda_uniform_scale(double *lambda, double left, double right){
   *lambda = left + *lambda * (right - left);
}

__global__ void cuda_bound_vector(int n, double *A, double *B,
                                  double *candidate, double *theta) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    A[i] = -candidate[i] / theta[i];
    B[i] = (1 - candidate[i]) / theta[i];
  }
}

__global__ void cuda_bound_determine(int n, double *left, double *right,
                                     double *A, double *B, double *theta) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    left[i] = A[i] * (theta[i] > 0) + B[i] * (theta[i] < 0);
    right[i] = A[i] * (theta[i] < 0) + B[i] * (theta[i] > 0);
  }
}

__global__ void cuda_all_positive(int n, double *a, bool *result) {
  *result = true;
  for (int i = 0; i < n; ++i)
    *result = *result && (a[i] > 0);
}

template <int SIGN, int BLKSZ>
__global__ void cuda_max_reduce(const double *d_data, const int d_len,
                                double *extrm_val) {
  volatile __shared__ double smem[BLKSZ];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  // starting index for each block to begin loading the input data into shared
  // memory
  const int bid_sidx = bid * BLKSZ;

// load the input data to smem, with padding if needed. each thread handles 2
// elements
#pragma unroll
  for (int i = 0; i < 2; i++) {
    // get the index for the thread to load into shared memory
    const int tid_idx = 2 * tid + i;
    const int ld_idx = bid_sidx + tid_idx;
    if (ld_idx < (bid + 1) * BLKSZ && ld_idx < d_len)
      smem[tid_idx] = d_data[ld_idx];
    else
      smem[tid_idx] = -SIGN * DBL_MAX;

    __syncthreads();
  }

  // run the reduction per-block
  for (unsigned int stride = BLKSZ / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] =
          ((SIGN * smem[tid] > SIGN * smem[tid + stride]) ? smem[tid]
                                                          : smem[tid + stride]);
    }
    __syncthreads();
  }

  // write the per-block result out from shared memory to global memory
  extrm_val[bid] = smem[0];
}

// assume we have d_data as a device pointer with our data, of length data_len
template <int SIGN, int elem_per_thread = 2, int thread_per_block = 256>
__host__ double cuda_find_max(const double *d_data, const int data_len) {
  // in your host code, invoke the kernel with something along the lines of:
  const int BLKSZ = elem_per_thread *
                    thread_per_block; // number of elements to process per block
  const int blocks_per_grid = ceil((float)data_len / (BLKSZ));

  double *d_extrm;
  cudaMallocManaged((void **)&d_extrm, sizeof(double) * blocks_per_grid);

  cuda_max_reduce<SIGN, BLKSZ>
      <<<blocks_per_grid, thread_per_block>>>(d_data, data_len, d_extrm);

  cudaDeviceSynchronize();

  double result = d_extrm[0];
  for (int i = 1; i < blocks_per_grid; i++) 
     result = ((SIGN * d_extrm[i] > SIGN * result) ? d_extrm[i] : result);

  cudaFree(d_extrm);

  return result;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat sampleCC(arma::field<arma::mat> A, NumericVector initial,
                   int n_replicates, double mu = 0.0, double sigma = 1.0,
                   int n_iter = 1.0e+5, int burn_in = 1.0e+3) {

  // Initialization
  int n = initial.size(); // sample size
  arma::vec initial_cdf = Rcpp::as<arma::vec>(wrap(pnorm(initial, mu, sigma)));
  arma::mat qsamples(n, n_replicates + burn_in, arma::fill::zeros);
  
  // GPU block and thread layout
  const int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  // Create pseudo-random number generator
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  // CUDA operations handlers
  cublasHandle_t handle;
  cublasStatus_t statHandle = cublasCreate(&handle);

  // Declaring GPU objects
  double *lambda, *norm2;
  double *candidateN, *candidateO, *candidateQ, *thetaV, *matrixCUDA,
      *resultCUDA, *cdtCUDA;
  double *boundA, *boundB, *leftV, *rightV;
  bool *bCUDA;

  // Resource allocation
  cudaMalloc(&lambda, sizeof(double));
  cudaMalloc(&norm2, sizeof(double));

  cudaMalloc(&candidateN, n * sizeof(double));
  cudaMalloc(&candidateO, n * sizeof(double));
  cudaMalloc(&candidateQ, n * sizeof(double));
  cudaMalloc(&thetaV, n * sizeof(double));
  cudaMalloc(&matrixCUDA, n * n * A.n_elem * sizeof(double));
  cudaMalloc(&resultCUDA, n * A.n_elem * sizeof(double));
  cudaMalloc(&cdtCUDA, A.n_elem * sizeof(double));

  cudaMalloc(&boundA, n * sizeof(double));
  cudaMalloc(&boundB, n * sizeof(double));
  cudaMalloc(&leftV, n * sizeof(double));
  cudaMalloc(&rightV, n * sizeof(double));

  cudaMalloc(&bCUDA, sizeof(bool));

  // Initialization of candidateN
  cudaMemcpy(candidateN, initial_cdf.memptr(), n * sizeof(double),
             cudaMemcpyHostToDevice);
  
  // Sequentially transfer constraint matrix to GPU
  arma::mat transA(n, n, arma::fill::zeros);
  for (int r = 0; r < A.n_elem; ++r) {
    transA = trans(A(r));
    cudaMemcpy(matrixCUDA + r * n * n * sizeof(double), transA.memptr(),
               n * n * sizeof(double), cudaMemcpyHostToDevice);
  }

  // Set these constants so we get a simple matrix multiply with cublasDgemm
  double alpha = 1.0;
  double beta = 0.0;
  
  bool bHOST;

  for (int s = 0; s < (n_replicates + burn_in); ++s) {

    // Step update
    cudaMemcpy(candidateO, candidateN, n * sizeof(double),
               cudaMemcpyDeviceToDevice);
    
    // Sample an n-dimensional vector from the unit sphere
    curandGenerateUniformDouble(gen, thetaV, n);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, n, &alpha, thetaV, 1,
                thetaV, n, &beta, norm2, 1);
    
    cuda_vector_normalize<<<numBlocks, blockSize>>>(n, norm2, thetaV);
    
    // Determine interval sampling boundaries for lambda
    cuda_bound_vector<<<numBlocks, blockSize>>>(n, boundA, boundB, candidateO,
                                                thetaV);
    cuda_bound_determine<<<numBlocks, blockSize>>>(n, leftV, rightV, boundA,
                                                   boundB, thetaV);
    
    double leftQ = cuda_find_max<1, 2, blockSize>(leftV, n);
    double rightQ = cuda_find_max<-1, 2, blockSize>(rightV, n);
   
    for (int iter = 0; iter < n_iter; ++iter) {
      if (iter == n_iter)
        stop("The quadratic constraints cannot be satisfied");

      curandGenerateUniformDouble(gen, lambda, 1);
      cuda_uniform_scale<<<1, 1>>>(lambda, leftQ, rightQ);

      cuda_affine_trans<<<numBlocks, blockSize>>>(lambda, n, candidateO,
                                                  thetaV, candidateN);
      cuda_inverse_cdf<<<numBlocks, blockSize>>>(n, candidateQ, candidateN, mu,
                                                 sigma);
      
      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n * A.n_elem, 1, n, &alpha,
                  matrixCUDA, n, candidateQ, n, &beta, resultCUDA,
                  n * A.n_elem);

      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, A.n_elem, 1, n, &alpha,
                  resultCUDA, n, candidateQ, n, &beta, cdtCUDA, A.n_elem);
       
      cuda_all_positive<<<1, 1>>>(A.n_elem, cdtCUDA, bCUDA);

      cudaDeviceSynchronize();
      
      cudaMemcpy(&bHOST, bCUDA, sizeof(bool),
                   cudaMemcpyDeviceToHost);
      if (bHOST) {
        cudaMemcpy(qsamples.colptr(s), candidateQ, n * sizeof(double),
                   cudaMemcpyDeviceToHost);
        break;
      }
    }
  }

  // CUDA free GPU objects
  cudaFree(lambda);
  cudaFree(norm2);

  cudaFree(candidateN);
  cudaFree(candidateO);
  cudaFree(candidateQ);
  cudaFree(thetaV);
  cudaFree(matrixCUDA);
  cudaFree(resultCUDA);
  cudaFree(cdtCUDA);

  cudaFree(boundA);
  cudaFree(boundB);
  cudaFree(leftV);
  cudaFree(rightV);
  cudaFree(bCUDA);  

  return qsamples.cols(burn_in, n_replicates + burn_in - 1);
}
