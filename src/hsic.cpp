#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

//' Computes the HSIC criterion for two given kernels
//'
//' The Hilbert-Schmidt Independence Criterion (HSIC) is a measure of independence
//' between two random variables. If characteristic kernels are used for both
//' variables, the HSIC is zero iff the variables are independent. In this
//' function, we implement an unbiased estimator for the HSIC measure. Specifically,
//' for two positive-definite kernels \eqn{K} and \eqn{L} and a sample size
//' \eqn{n}, the unbiased HSIC estimator is:
//' \deqn{HSIC(K, L) = \frac{1}{n(n-3)} \left[trace(KL) + \frac{1^\top K11^\top L 1}{(n-1)(n-2)}- \frac{2}{n-2}1^\top KL\right]}
//'
//' @param K first kernel similarity matrix
//' @param L second kernel similarity matrix
//'
//' @return an unbiased estimate of the HSIC measure.
//'
//' @references Song, L., Smola, A., Gretton, A., Borgwardt, K., & Bedo, J.
//' (2007). Supervised Feature Selection via Dependence Estimation.
//' https://doi.org/10.1145/1273496.1273600
//'
//' @examples
//' n <- 50
//' p <- 20
//' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
//' Y <- matrix(rnorm(n*p), nrow = n, ncol = p)
//' K <-  X %*% t(X) / p
//' L <-  Y %*% t(Y) / p
//' uHSIC <- HSIC(K, L)
//'
//' @export
// [[Rcpp::export]]
double HSIC(arma::mat K, arma::mat L)
{
    
    double n = (double) K.n_rows;
    
    arma::mat KK = K - arma::diagmat(K);
    arma::mat LL = L - arma::diagmat(L);
    arma::colvec onecol = arma::ones(n);
    arma::rowvec onerow = onecol.t();
    
    double traceKL = 0;
    for (int i = 0; i < n; i++) {
        traceKL += arma::as_scalar(KK.row(i) * LL.col(i)); 
    }
    
    double hsic = (traceKL + arma::accu(KK) * arma::accu(LL) / ((n - 1) * (n - 2))
                       - 2 * arma::dot(onerow * KK, LL * onecol) / (n - 2)) / (n * (n - 3));
    
    return hsic;
}

//' Determines the quadratic form of the HSIC unbiased estimator
//'
//' For a linear kernel of the outcome \eqn{L = Y^\top Y}, the unbiased HSIC
//' estimator implemented in \code{\link{HSIC}} can be expressed as a quadratic
//' form of the outcome \eqn{Y} i.e. \eqn{HSIC(K, L) = Y^\top Q(K) Y}. Here,
//' the matrix \eqn{Q} only depends on the kernel similarity matrix \eqn{K}.
//'
//' @param K kernel similarity matrix
//'
//' @return the matrix of the HSIC estimator quadratic form
//'
//' @examples
//' n <- 50
//' p <- 20
//' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
//' K <-  X %*% t(X) / p
//' Q <- quadHSIC(K)
//' @export
// [[Rcpp::export]]
arma::mat quadHSIC(arma::mat K)
{
    double n = (double) K.n_rows;
    arma::colvec onecol = arma::ones(n);
    arma::rowvec onerow = onecol.t();
    
    arma::mat Q = K - arma::diagmat(K) +
        arma::accu(K - arma::diagmat(K)) *
        (arma::ones(size(K)) - arma::eye(size(K)))/ ((n - 1) * (n - 2)) -
        (2 / (n -2)) * (onecol * (onerow * K) - arma::diagmat(onecol * (onerow * K)) -
        onecol * (onerow * arma::diagmat(K)) +
        arma::diagmat(onecol * (onerow * arma::diagmat(K))));
    
    return Q;
}
