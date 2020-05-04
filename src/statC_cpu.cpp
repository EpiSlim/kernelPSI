#define ARMA_ALLOW_FAKE_GCC

#include <RcppArmadillo.h>
// #include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

#include "hsic.h"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

double statC(arma::vec sample, arma::mat replicates,
              arma::field<arma::mat> K) {

  //stop("Trying to access a GPU function");

  return 0.0;
}

