#include <RcppArmadillo.h>
#include "hsic.h"
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

//' selects a fixed number of kernels which are most associated with the
//' outcome kernel. 
//' 
//' @param K list of kernel similarity matrices 
//' @param L kernel similarity matrix for the outcome
//' @param mKernels number of kernels to be selected
//' 
//' @return an integer vector containing the indices of the selected kernels 
//' 
//' @examples
//' n <- 50
//' p <- 20
//' K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
//' L <- matrix(rnorm(n*p), nrow = n, ncol = p)
//' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
//' L <-  L %*% t(L) / p 
//' selection <- FOHSIC(K, L, 2)
//' 
//' @export
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

//' models the forward selection event of a fixed number of kernels as a 
//' succession of quadratic constraints
//' 
//' @param K list kernel similarity matrices
//' @param select integer vector containing the indices of the selected kernels
//' 
//' @return list of matrices modeling the quadratic constraints of the 
//' selection event 
//' 
//' @examples
//' n <- 50
//' p <- 20
//' K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
//' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
//' listQ <- forwardQ(K, select = c(4, 1))
//' @export
// [[Rcpp::export]]
arma::field<arma::mat> forwardQ(arma::field<arma::mat> K, IntegerVector select)
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
