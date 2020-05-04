#define ARMA_ALLOW_FAKE_GCC

#include <RcppArmadillo.h>
//#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// CUDA headers
#include "cublas_v2.h"
#include <cuda.h>
#include <cuda_runtime.h>

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat sampleC(arma::field<arma::mat> A, NumericVector initial,
                   int n_replicates, double mu = 0.0, double sigma = 1.0,
                   int n_iter = 1.0e+5, int burn_in = 1.0e+3) {

  int n = initial.size();
  arma::mat qsamples(n, n_replicates + burn_in, arma::fill::zeros);
  arma::vec candidateO(n), candidateQ(n),
      candidateN = Rcpp::as<arma::vec>(wrap(pnorm(initial, mu, sigma)));
  arma::vec cdt(A.n_elem);

  // Randomly sample in the sphere unit
  arma::vec thetaV(n);
  arma::mat theta(n, n_replicates + burn_in, arma::fill::randn);
  theta = arma::normalise(theta, 2, 0);

  // Rejection sampling
  arma::vec::iterator l;
  arma::vec boundA, boundB;
  arma::mat matA(n * A.n_elem, n);
  for (int r = 0; r < A.n_elem; ++r) {
    matA(n * r, 0, size(A(r))) =
        A(r); // Regrouping the list of matrices in a single GPU matrix
  }

  // Declaring GPU objects
  double *matrixCUDA, *prodCUDA, *vectorCUDA, *cdtCUDA;

  // Resource allocation
  cudaMalloc(&vectorCUDA, n * sizeof(double));
  cudaMalloc(&matrixCUDA, n * n * A.n_elem * sizeof(double));
  cudaMalloc(&prodCUDA, n * A.n_elem * sizeof(double));
  cudaMalloc(&cdtCUDA, A.n_elem * sizeof(double));

  // Transfer data to GPU
  cudaMemcpy(matrixCUDA, matA.memptr(), n * n * A.n_elem * sizeof(double),
             cudaMemcpyHostToDevice);

  // Set these constants so we get a simple matrix multiply with cublasDgemm
  double alpha = 1.0;
  double beta = 0.0;

  // CUDA operations handlers
  cublasHandle_t handle;
  cublasStatus_t statHandle = cublasCreate(&handle);

  for (int s = 0; s < (n_replicates + burn_in); ++s) {

    candidateO = candidateN;
    thetaV = theta.col(s);

    boundA = -(candidateO / thetaV);
    boundB = (1 - candidateO) / thetaV;

    double leftQ = std::max(boundA.elem(arma::find(thetaV > 0)).max(),
                            boundB.elem(arma::find(thetaV < 0)).max());
    double rightQ = std::min(boundA.elem(arma::find(thetaV < 0)).min(),
                             boundB.elem(arma::find(thetaV > 0)).min());

    for (int iter = 0; iter < n_iter; ++iter) {
      if (iter == n_iter)
        stop("The quadratic constraints cannot be satisfied");

      double lambda = runif(1, leftQ, rightQ)[0];
      candidateN = candidateO + lambda * thetaV;
      candidateQ = Rcpp::as<arma::vec>(
          wrap(qnorm(as<NumericVector>(wrap(candidateN)), mu, sigma)));
      
      cudaMemcpy(vectorCUDA, candidateQ.memptr(), n * sizeof(double),
                 cudaMemcpyHostToDevice);
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n * A.n_elem, 1, n, &alpha,
                  matrixCUDA, n * A.n_elem, vectorCUDA, n, &beta, prodCUDA,
                  n * A.n_elem);
      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, A.n_elem, 1, n, &alpha,
                  prodCUDA, n, vectorCUDA, n, &beta, cdtCUDA, A.n_elem);
      cudaDeviceSynchronize();
      cudaMemcpy(cdt.memptr(), cdtCUDA, A.n_elem * sizeof(double),
                 cudaMemcpyDeviceToHost);

      if (all(cdt >= 0)) {
        qsamples.col(s) = candidateQ;
        break;
      }
    }
  }

  // CUDA free GPU objects
  cudaFree(vectorCUDA);
  cudaFree(matrixCUDA);
  cudaFree(prodCUDA);
  cudaFree(cdtCUDA);

  cublasDestroy(handle);

  return qsamples.cols(burn_in, n_replicates + burn_in - 1);
}
