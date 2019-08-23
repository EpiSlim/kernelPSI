#define ARMA_ALLOW_FAKE_GCC

#define VIENNACL_WITH_CUDA
#define VIENNACL_WITH_OPENMP
#define VIENNACL_WITH_ARMADILLO 1

#include <ctime>
#include <ratio>
#include <chrono>
using namespace std::chrono;

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

arma::mat sampleT(arma::field<arma::mat> A, NumericVector initial, int n_replicates,
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
    arma::vec boundA, boundB;
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

    double time_prod = 0, time_loop = 0, time_out = 0;
    high_resolution_clock::time_point prodt1, prodt2, loopt1, loopt2, outt1, outt2;

    int r;
    outt1 = high_resolution_clock::now();
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

            double lambda = runif(1, leftQ, rightQ)[0];
            candidateN = candidateO + lambda * thetaV;
            candidateQ = Rcpp::as<arma::vec>(wrap(qnorm(as<NumericVector>(wrap(candidateN)), mu, sigma)));

            loopt1 = high_resolution_clock::now();
            viennacl::copy(candidateQ, vectorCL);
            loopt2 = high_resolution_clock::now();
            time_loop += duration_cast<duration<double>>(loopt2 - loopt1).count();

            prodt1 = high_resolution_clock::now();
            resultCL = viennacl::linalg::prod(matrixCL, vectorCL);
            prodt2 = high_resolution_clock::now();
            time_prod += duration_cast<duration<double>>(prodt2 - prodt1).count();


            baseCL = viennacl::matrix_base<double> (resultCL.handle(),
                                                    A.n_elem, 0, 1, A.n_elem,
                                                    n, 0, 1, n,
                                                    true);

            prodCL = baseCL;


            cdtCL = viennacl::linalg::prod(prodCL, vectorCL);

            if (viennacl::linalg::min(cdtCL) >= 0) {
                qsamples.col(s) = candidateQ;
                break;
            }

        }


    }
    outt2 = high_resolution_clock::now();
    time_out += duration_cast<duration<double>>(outt2 - outt1).count();

    Rprintf("time prod = %f secs\n", time_prod);
    Rprintf("time loop = %f secs\n", time_loop);
    Rprintf("time out = %f secs\n", time_out);

    return qsamples.cols(burn_in, n_replicates + burn_in - 1);
}
