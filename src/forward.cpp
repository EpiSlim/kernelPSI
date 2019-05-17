#include <RcppArmadillo.h>
#include "hsic.h"
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
IntegerVector FOHSIC(arma::field<arma::mat> K, arma::mat L, int mKernels = 1)
{

    arma::vec hvec(K.n_elem);
    arma::vec::iterator h;
    int r;

    for (h = hvec.begin(), r = 0; h != hvec.end(); ++h, ++r)
    {
        *h = HSIC(K(r), L);
    }

    arma::uvec selection = sort_index(hvec, "descend") + 1;

    return wrap(selection(arma::span(0, mKernels - 1)));

}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//

/*** R
nconstraints <- 40
n <- 20
K <- replicate(nconstraints, matrix(rnorm(n*n), ncol = n, nrow = n), simplify = FALSE)
L <- matrix(rnorm(n*n), ncol = n, nrow = n)
aa <- FOHSIC(K, L, nconstraints)
*/
