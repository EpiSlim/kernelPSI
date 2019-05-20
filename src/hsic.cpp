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
