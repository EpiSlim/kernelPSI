#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// #define VIENNACL_WITH_CUDA
// #define VIENNACL_WITH_OPENCL
// #define VIENNACL_WITH_OPENMP
#define VIENNACL_WITH_ARMADILLO 1


// ViennaCL headers
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/forwards.h"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/inner_prod.hpp"  


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat sampleC(arma::field<arma::mat> A, NumericVector initial, int n_replicates,
                  double mu = 0.0, double sigma = 1.0,
                  int n_iter = 1.0e+5, int burn_in = 1.0e+3)
{

    int n = initial.size();
    arma::mat qsamples(n, n_replicates + burn_in, arma::fill::zeros);
    arma::mat candidates(n, n_replicates + burn_in + 1, arma::fill::zeros);
    candidates.col(0) = Rcpp::as<arma::vec>(wrap(pnorm(initial, mu, sigma)));
    arma::vec candidateO(n), candidateN = Rcpp::as<arma::vec>(wrap(pnorm(initial, mu, sigma)));

    // Randomly sample in the sphere unit
    arma::mat theta(n, n_replicates + burn_in, arma::fill::randn);
    theta = normalise(theta, 2, 0);

    // Rejection sampling
    arma::vec cdt(A.n_elem);
    arma::vec::iterator l;
    arma::vec boundA, boundB;
    
    // Declaring GPU objects
    viennacl::vector<double> vectorCL(n), resultCL(n);
    viennacl::matrix<double, viennacl::column_major> matrixCL(n, n*A.n_elem); 
    for (int r = 0; r < A.n_elem; ++r){
        viennacl::copy(A(r), project(matrixCL, viennacl::range(0, n), viennacl::range(n*r, n*(r+1)))); // Regrouping the list of matrices in a single GPU matrix
    }
    
    
    int r;
    for (int s = 0; s < (n_replicates + burn_in); ++s)
    {
        candidateO = candidateN;
        
        boundA = -(candidateO/theta.col(s));
        boundB = (1 - candidateO)/theta.col(s);
        double leftQ = std::max(boundA.elem(arma::find(theta.col(s) > 0)).max(),
                                boundB.elem(arma::find(theta.col(s) < 0)).max());
        double rightQ = std::min(boundA.elem(arma::find(theta.col(s) < 0)).min(),
                                 boundB.elem(arma::find(theta.col(s) > 0)).min());
        
        for (int iter = 0; iter < n_iter; ++iter)
        {
            if (iter == n_iter) stop("The quadratic constraints cannot be satisfied");
            double lambda = runif(1, leftQ, rightQ)[0];
            candidateN = candidateO + lambda * theta.col(s);
            qsamples.col(s) = Rcpp::as<arma::vec>(wrap(qnorm(as<NumericVector>(wrap(candidateN)), mu, sigma)));
            copy(qsamples.col(s), vectorCL);
            for(l = cdt.begin(), r = 0; l != cdt.end(); ++l, ++r)
            {
                resultCL = viennacl::linalg::prod(project(matrixCL, viennacl::range(0, n), viennacl::range(n*r, n*(r+1)), vectorCL);
                *l = viennacl::linalg::inner_prod(vectorCL, resultCL); 
            }
            if (all(cdt >= 0)) {
                break;
            }

        }
    }

    return qsamples.cols(burn_in, n_replicates + burn_in - 1);
}
