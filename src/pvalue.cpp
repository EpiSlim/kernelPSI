#define ARMA_ALLOW_FAKE_GCC

#include <RcppArmadillo.h>
// #include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

#include "hsic.h"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

double pvalue(arma::vec sample, arma::mat replicates, arma::field<arma::mat> K) {

  int n = sample.size();
  arma::mat Ksum(n, n, arma::fill::zeros), HSICmat(n, n, arma::fill::zeros);

  // Computing the quadratic form of the selected kernels
  for (unsigned int r = 0; r < K.n_elem; ++r) {
    Ksum += K(r);
  }
  HSICmat = quadHSIC(Ksum);
  
  // Computing the statistics of the replicates
  arma::rowvec stat =
      arma::ones(replicates.n_rows).t() * ((HSICmat * replicates) % replicates);
  // Computing the statistic of original sample
  double statS = arma::as_scalar(sample.t() * HSICmat * sample);

  // Determine the p-value
  double pvalue = arma::sum(stat.t() > statS) / (double)replicates.n_cols;

  return pvalue;
}
