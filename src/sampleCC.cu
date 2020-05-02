#define ARMA_ALLOW_FAKE_GCC

#include <chrono>
#include <ctime>
#include <ratio>
using namespace std::chrono;

#include <RcppArmadillo.h>
//#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// CUDA headers
#include "cublas_v2.h"
#include "curand.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "maxmin.cuh"

// CUDA kernels
__global__ void cuda_affine_trans(double lambda, int n, double *a, double *b,
                                  double *c) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    c[i] = a[i] + lambda * b[i];
}

__global__ void cuda_inverse_cdf(int n, double *a, double *b, double mu,
                                 double sigma) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    a[i] = normcdfinv((b[i] - mu) / sigma);
}

__global__ void cuda_vector_normalize(int n, double norm2, double *a) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    a[i] /= norm2;
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

__global__ void host_all_positive(int n, double *a, bool *result) {
  *result = 1;
  for (int i = 0; i < n; ++i)
    *result *= a[i] > 0;
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

  // do not forget to intialiaze candidateN
  cudaMemcpy(candidateN, initial_cdf, n * sizeof(double),
             cudaMemcpyHostToDevice);

  // Sequentially transfer constraint matrix to GPU
  for (int r = 0; r < A.n_elem; ++r) {
    cudaMemcpy(matrixCUDA + r * n * n * sizeof(double), trans(A(r)).memptr(),
               n * n * sizeof(double), cudaMemcpyHostToDevice);
  }

  // GPU block and thread layout
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  // Create pseudo-random number generator
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  // CUDA operations handlers
  cublasHandle_t handle;
  cublasStatus_t statHandle = cublasCreate(&handle);

  // Set these constants so we get a simple matrix multiply with cublasDgemm
  double alpha = 1.0;
  double beta = 0.0;

  for (int s = 0; s < (n_replicates + burn_in); ++s) {

    // Step update
    cudaMemcpy(candidateO, candidateN, n * sizeof(double),
               cudaMemcpyDeviceToDevice);

    // Sample an n-dimensional vector from the unit sphere
    curandGenerateUniformDouble(gen, thetaV, n);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, n, &alpha, thetaV, 1,
                thetaV, n, &beta, norm2, 1);
    cuda_vector_normalize<<<numBlocks, blockSize>>>(n, *norm2, thetaV);

    // Determine interval sampling boundaries for lambda
    cuda_bound_vector<<<numBlocks, blockSize>>>(n, boundA, boundB, candidateO,
                                                thetaV);
    cuda_bound_determine<<<numBlocks, blockSize>>>(n, leftV, rightV, boundA,
                                                   boundB, thetaV);

    double leftQ = cu_find_max<1, 2, blockSize>(leftV, n);
    double rightQ = cu_find_max<-1, 2, blockSize>(rightV, n);

    for (int iter = 0; iter < n_iter; ++iter) {
      if (iter == n_iter)
        stop("The quadratic constraints cannot be satisfied");

      curandGenerateUniformDouble(gen, lambda, 1);
      *lambda = leftQ + *lambda * (rightQ - leftQ);

      cuda_affine_trans<<<numBlocks, blockSize>>>(*lambda, n, candidateO,
                                                  thetaV, candidateN);
      cuda_inverse_cdf<<<numBlocks, blockSize>>>(n, candidateQ, candidateN, mu,
                                                 sigma);

      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n * A.n_elem, 1, n, &alpha,
                  matrixCUDA, n, candidateQ, n, &beta, resultCUDA,
                  n * A.n_elem);

      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, A.n_elem, 1, n, &alpha,
                  resultCUDA, n, candidateQ, n, &beta, cdtCUDA, A.n_elem);

      host_all_positive<<<1, 1>>>(A.n_elem, cdtCUDA, bCUDA);

      cudaDeviceSynchronize();

      if (*bCUDA) {
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

  return qsamples.cols(burn_in, n_replicates + burn_in - 1);
}
