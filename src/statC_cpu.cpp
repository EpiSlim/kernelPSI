#define ARMA_ALLOW_FAKE_GCC

#include <RcppArmadillo.h>
// #include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

#include "hsic.h"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double statC2(arma::vec sample, arma::mat replicates, arma::field<arma::mat> K) {

  int n = sample.size();
  arma::mat Ksum(n, n, arma::fill::zeros), HSICmat(n, n, arma::fill::zeros);

  // Computing the quadratic form of the selected kernels
  for (int r = 0; r < K.n_elem; ++r) {
    Ksum += K(r);
  }
  HSICmat = quadHSIC(Ksum);

  // Computing the statistics of the replicates
  arma::vec stat =
      arma::ones(replicates.n_rows).t() * ((Ksum * replicates) % replicates);

  // Computing the statistic of original sample
  double statS = sample.t() * Ksum * sample;

  // Compute p-value
  double pvalue = arma::sum(stat > statS) / (double)replicates.n_cols;

  return pvalue;
}
