#define ARMA_ALLOW_FAKE_GCC

#include <RcppArmadillo.h>
// #include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// CUDA & Thrust headers
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// Timing headers
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <cstdlib>
#define USECPSEC 1000000ULL

long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat sampleQ(arma::field<arma::mat> A, NumericVector initial, int n_replicates,
                  double mu = 0.0, double sigma = 1.0,
                  int n_iter = 1.0e+5, int burn_in = 1.0e+3)
{

    int n = initial.size();
    arma::mat qsamples(n, n_replicates + burn_in, arma::fill::zeros);
    arma::mat candidates(n, n_replicates + burn_in + 1, arma::fill::zeros);
    arma::vec candidateO(n), candidateQ(n), candidateN = Rcpp::as<arma::vec>(wrap(pnorm(initial, mu, sigma)));

    // Randomly sample in the sphere unit
    arma::vec thetaV(n);
    arma::mat theta(n, n_replicates + burn_in, arma::fill::randn);
    theta = arma::normalise(theta, 2, 0);

    // Rejection sampling
    arma::vec::iterator l;
    arma::vec boundA, boundB, cdt(n);
    arma::mat matA(n*A.n_elem, n);
    for (int r = 0; r < A.n_elem; ++r){
        matA(n*r, 0, size(A(r))) = A(r); // Regrouping the list of matrices in a single GPU matrix
    }

    // Declaring GPU objects
    double *matrixT;
    cudaMalloc(&matrixT, n*n*A.n_elem * sizeof(double));
    cudaMemcpy(matA.begin(), matrixT, n*n*A.n_elem * sizeof(double),cudaMemcpyHostToDevice);

    double *resultT;
    cudaMalloc(&resultT, n*A.n_elem * sizeof(double));

    double *vectorT;
    cudaMalloc(&vectorT, n * sizeof(double));

    double *cdtT;
    cudaMalloc(&cdtT, A.n_elem * sizeof(double));

    cublasHandle_t h;
    cublasCreate(&h);
    double alpha = 1.0, beta = 0.0;

    int r;
    for (int s = 0; s < (n_replicates + burn_in); ++s)
    {


        candidateO = candidateN;
        thetaV = theta.col(s);

        boundA = -(candidateO/thetaV);
        boundB = (1 - candidateO)/thetaV;

        double leftQ = std::max(boundA.elem(arma::find(thetaV > 0)).max(),
                                boundB.elem(arma::find(thetaV < 0)).max());
        double rightQ = std::min(boundA.elem(arma::find(thetaV < 0)).min(),
                                 boundB.elem(arma::find(thetaV > 0)).min());


        for (int iter = 0; iter < n_iter; ++iter)
        {
            if (iter == n_iter) stop("The quadratic constraints cannot be satisfied");
            Rprintf("iteration == %d\n", iter);

            double lambda = runif(1, leftQ, rightQ)[0];
            candidateN = candidateO + lambda * thetaV;
            candidateQ = Rcpp::as<arma::vec>(wrap(qnorm(as<NumericVector>(wrap(candidateN)), mu, sigma)));

            cudaMemcpy(candidateQ.begin(), vectorT, n*sizeof(double),cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            long long dt = dtime_usec(0);
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n*A.n_elem, 1, n, &alpha,
                        matrixT, n*A.n_elem, vectorT, n, &beta,
                        resultT, n*A.n_elem);
            cudaDeviceSynchronize();
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 1, A.n_elem, n, &alpha,
                        vectorT, 1, resultT, n, &beta,
                        cdtT, 1);
            cudaDeviceSynchronize();
            dt = dtime_usec(dt);
            std::cout << "GPU computation time: " << dt/(float)USECPSEC << "s" << std::endl;
            cudaMemcpy(cdt.begin(), cdtT, A.n_elem * sizeof(double),cudaMemcpyDeviceToHost);

            if (all(cdt >= 0)) {
                qsamples.col(s) = candidateQ;
                break;
            }

        }
    }

    cudaFree(matrixT);
    cudaFree(resultT);
    cudaFree(vectorT);
    cudaFree(cdtT);

    return qsamples.cols(burn_in, n_replicates + burn_in - 1);
}
