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
IntegerVector adaFOHSIC(arma::field<arma::mat> K, arma::mat L, int mKernels = 1)
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

// [[Rcpp::export]]
arma::field<arma::mat> adaQ(arma::field<arma::mat> K, IntegerVector select)
{
  arma::field<arma::mat> constraintQ(K.n_elem - 1);
  int s;
  if (select.size() > 1)
  {
    for (s = 0; s != (select.size() - 1); s++)
    {
      constraintQ(s) = quadHSIC(K(select(s) - 1)) - quadHSIC(K(select(s + 1) - 1));
    }
    
  }
  
  int r = 0;
  for (s = 0; s != K.n_elem; s++)
  {
    if (!(std::find(select.begin(), select.end(), s + 1) != select.end()))
    {
      constraintQ(select.size() + r - 1) = quadHSIC(K(select(select.size() - 1) - 1)) - quadHSIC(K(s));
      r += 1;
    }
  }
  return constraintQ;
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
aa <- FOHSIC(K, L, nconstraints %/% 2)
bb <- forwardQ(K, aa)
*/
