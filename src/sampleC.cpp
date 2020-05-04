#define ARMA_ALLOW_FAKE_GCC

#include <chrono>
#include <ctime>
#include <ratio>
using namespace std::chrono;

#include <RcppArmadillo.h>
// #include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

# include "sampleH.hpp"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat sampleC(arma::field<arma::mat> A, NumericVector initial,
                   int n_replicates, double mu = 0.0, double sigma = 1.0,
                   int n_iter = 1.0e+5, int burn_in = 1.0e+3) {

  return sampleH(A, initial, n_replicates, mu, sigma, n_iter, burn_in);
}
