#define ARMA_ALLOW_FAKE_GCC

#define VIENNACL_WITH_CUDA
#define VIENNACL_WITH_OPENMP
#define VIENNACL_WITH_ARMADILLO 1

// Timing headers
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <cstdlib>
#define USECPSEC 1000000ULL

long long ftime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// ViennaCL headers
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/forwards.h"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/maxmin.hpp"


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat sampleC(arma::field<arma::mat> A, NumericVector initial, int n_replicates,
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
    arma::vec boundA, boundB, cdt(A.n_elem), leftV(n), rightV(n);
    arma::mat matA(n*A.n_elem, n);
    for (int r = 0; r < A.n_elem; ++r){
        matA(n*r, 0, size(A(r))) = A(r); // Regrouping the list of matrices in a single GPU matrix
    }

    // Declaring GPU objects
    viennacl::matrix_base<double> baseCL(A.n_elem, n, true);
    viennacl::vector<double> vectorCL(n), resultCL(n*A.n_elem), cdtCL(A.n_elem);
    viennacl::matrix<double, viennacl::column_major> matrixCL(n*A.n_elem, n);
    viennacl::matrix<double, viennacl::row_major> prodCL(A.n_elem, n);
    copy(matA, matrixCL);

    int r;
    for (int s = 0; s < (n_replicates + burn_in); ++s)
    {


        candidateO = candidateN;
        thetaV = theta.col(s);

        boundA = -(candidateO/thetaV);
        boundB = (1 - candidateO)/thetaV;

        //leftV = boundA * (thetaV > 0) + boundB * (thetaV < 0);
        //rightV = boundA * (thetaV < 0) + boundB * (thetaV > 0);

        //double leftQ = leftV.max();
        //double rightQ = rightV.min();

        //long long dt = ftime_usec(0);
        double leftQ = std::max(boundA.elem(arma::find(thetaV > 0)).max(),
                                boundB.elem(arma::find(thetaV < 0)).max());
        double rightQ = std::min(boundA.elem(arma::find(thetaV < 0)).min(),
                                 boundB.elem(arma::find(thetaV > 0)).min());
        //dt = ftime_usec(dt);
        //std::cout << "bounds time: " << dt/(float)USECPSEC << "s" << std::endl;

        for (int iter = 0; iter < n_iter; ++iter)
        {
            if (iter == n_iter) stop("The quadratic constraints cannot be satisfied");

            double lambda = runif(1, leftQ, rightQ)[0];
            candidateN = candidateO + lambda * thetaV;
            candidateQ = Rcpp::as<arma::vec>(wrap(qnorm(as<NumericVector>(wrap(candidateN)), mu, sigma)));

            viennacl::copy(candidateQ, vectorCL);
            resultCL = viennacl::linalg::prod(matrixCL, vectorCL);
            baseCL = viennacl::matrix_base<double> (resultCL.handle(),
                                                    A.n_elem, 0, 1, A.n_elem,
                                                    n, 0, 1, n,
                                                    true);
            prodCL = baseCL;
            cdtCL = viennacl::linalg::prod(prodCL, vectorCL);
            viennacl::copy(cdtCL, cdt);

            if (all(cdt >= 0)) {
                qsamples.col(s) = candidateQ;
                break;
            }

        }


    }

    return qsamples.cols(burn_in, n_replicates + burn_in - 1);
}
