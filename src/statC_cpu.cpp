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
  for (unsigned int r = 0; r < K.n_elem; ++r) {
    Ksum += K(r);
  }
  HSICmat = quadHSIC(Ksum);
  double statS = arma::as_scalar(sample.t() * HSICmat * sample);
  Rcout << "First value" << statS << std::endl;
  Rcout << "hello 1" << std::endl;
  // Computing the statistics of the replicates
  arma::rowvec stat =
      arma::ones(replicates.n_rows).t() * ((HSICmat * replicates) % replicates);
  Rcout << "hello 2" << std::endl;
  // Computing the statistic of original sample
  // double statS = arma::as_scalar(sample.t() * Ksum * sample);
  // Compute p-value
  double pvalue = arma::sum(stat.t() > statS) / (double)replicates.n_cols;

  return pvalue;
}
