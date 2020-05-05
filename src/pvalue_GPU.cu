#define ARMA_ALLOW_FAKE_GCC

#include <RcppArmadillo.h>
// #include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

#include "hsic.h"

// CUDA headers
#include "cublas_v2.h"
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernels
__global__ void cuda_element_prod(int n, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    x[i] *= y[i];
}

__global__ void cuda_column_sum(int n, int p, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < p; i += stride) {
    x[i] = 0;
    for (int j = 0; j < n; ++j)
      x[i] += y[i * n + j];
  }
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

double pvalue(arma::vec sample, arma::mat replicates,
              arma::field<arma::mat> K) {

  // Compute the sum kernel
  int n = sample.size();
  arma::vec stat(replicates.n_cols);
  arma::mat Ksum(n, n, arma::fill::zeros);
  for (int r = 0; r < K.n_elem; ++r) {
    Ksum += K(r);
  }
  Ksum = quadHSIC(Ksum);

  // CUDA section
  cublasHandle_t handle;
  cublasStatus_t statHandle = cublasCreate(&handle);

  double *hsicCUDA, *replicatesCUDA, *prodCUDA, *sampleCUDA, *tmpCUDA,
      *statCUDA, *statS;

  // Allocate all our host-side (CPU) and device-side (GPU) data
  cudaMallocManaged(&hsicCUDA, n * n * sizeof(double));
  cudaMallocManaged(&replicatesCUDA,
                    replicates.n_rows * replicates.n_cols * sizeof(double));
  cudaMallocManaged(&sampleCUDA, n * sizeof(double));
  cudaMallocManaged(&statCUDA, replicates.n_cols * sizeof(double));
  cudaMallocManaged(&statS, sizeof(double));
  cudaMalloc(&prodCUDA, replicates.n_rows * replicates.n_cols * sizeof(double));
  cudaMalloc(&tmpCUDA, n * sizeof(double));

  // Copy data to CUDA objects
  cudaMemcpy(hsicCUDA, Ksum.memptr(), n * n * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(replicatesCUDA, replicates.memptr(),
             replicates.n_rows * replicates.n_cols * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(sampleCUDA, sample.memptr(), n * sizeof(double),
             cudaMemcpyHostToDevice);

  // Set these constants so we get a simple matrix multiply with cublasDgemm
  double alpha = 1.0;
  double beta = 0.0;

  // Computing the statistic for replicates
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, replicates.n_cols, n, &alpha,
              hsicCUDA, n, replicatesCUDA, replicates.n_rows, &beta, prodCUDA,
              n);

  int blockSize = 256;
  int numBlocks =
      (replicates.n_rows * replicates.n_cols + blockSize - 1) / blockSize;
  cuda_element_prod<<<numBlocks, blockSize>>>(
      replicates.n_rows * replicates.n_cols, prodCUDA, replicatesCUDA);
  cuda_column_sum<<<numBlocks, blockSize>>>(
      replicates.n_rows, replicates.n_cols, statCUDA, prodCUDA);

  // Computing statistic for original sample
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, n, &alpha, hsicCUDA, n,
              sampleCUDA, n, &beta, tmpCUDA, n);

  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, n, &alpha, tmpCUDA, 1,
              sampleCUDA, n, &beta, statS, 1);

  cudaDeviceSynchronize();

  cudaMemcpy(stat.memptr(), statCUDA, replicates.n_cols * sizeof(double),
             cudaMemcpyHostToHost);
   Rcout << "First value" << *statS << std::endl;
  // Compute p-value
  double pvalue = arma::sum(stat > *statS) / (double)replicates.n_cols;

  // Free resources
  cublasDestroy(handle);

  cudaFree(hsicCUDA);
  cudaFree(replicatesCUDA);
  cudaFree(prodCUDA);
  cudaFree(sampleCUDA);
  cudaFree(tmpCUDA);
  cudaFree(statCUDA);
  cudaFree(statS);

  return pvalue;
}

