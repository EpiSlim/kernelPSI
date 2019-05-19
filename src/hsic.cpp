#include <RcppArmadillo.h>
using namespace Rcpp;


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
double HSIC(arma::mat K, arma::mat L)
{

    double n = (double) K.n_rows;

    arma::mat KK = K - arma::diagmat(K);
    arma::mat LL = L - arma::diagmat(L);
    arma::mat KL = KK * LL;

    double hsic = (arma::trace(KL) + arma::accu(KK) * arma::accu(LL) / ((n - 1) * (n - 2))
                   - 2 * arma::accu(KL) / (n - 2)) / (n * (n - 3));

    return hsic;
}

// [[Rcpp::export]]
arma::mat quadHSIC(arma::mat K)
{
    double n = (double) K.n_rows;

     arma::mat Q = K - arma::diagmat(K) +
                  arma::accu(K - arma::diagmat(K)) *
                  (arma::ones(size(K)) - arma::eye(size(K)))/ ((n - 1) * (n - 2)) -
                  (2 / (n -2)) * (arma::ones(size(K)) * K - arma::diagmat(arma::ones(size(K)) * K) -
                                  arma::ones(size(K)) * arma::diagmat(K) +
                                  arma::diagmat(arma::ones(size(K)) * arma::diagmat(K)));

    return Q;
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//

/*** R
require("pracma")
n <- 20
K <- matrix(rnorm(n*n), ncol = n, nrow = n)
Y <- rnorm(n)
L <- Y %*% t(Y)
Qmat <- quadHSIC(K)

RHSIC <- function(K) {
  n <- dim(K)[[1]]
  Q <- (K - diag(diag(K)) +
    Reduce(`+`, K - diag(diag(K))) * (ones(n) - eye(n)) / ((n - 1) * (n - 2)) -
    (2 / (n - 2)) * (ones(n) %*% K - diag(diag(ones(n) %*% K)) -
      ones(n) %*% diag(diag(K)) + diag(diag(ones(n) %*% diag(diag(K))))))

  return(Q)
}

Rmat <- RHSIC(K)

Qmat[1:5, 1:5]
Rmat[1:5, 1:5]

*/
