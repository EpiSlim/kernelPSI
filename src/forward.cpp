#include <RcppArmadillo.h>
#include "hsic.h"
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

//' selects a fixed number of kernels which are most associated with the
//' outcome kernel.
//'
//' This function implements a forward algorithm for kernel selection. In the
//' first step, the kernel which maximizes the HSIC measure with the outcome
//' kernel \code{L} is selected. In the subsequent iterations, the kernel which,
//' combined with the selected kernels maximizes the HSIC measure is selected.
//' For the sum kernel combination rule, the forward algorithm can be
//' simplified. The kernels which maximize the HSIC measure with the kernel
//' \code{L} are selected in a descending order.
//'
//' \code{\link{FOHSIC}} implements the forward algorithm with a predetermined
//' number of kernels \code{mKernels}. If the exact number of causal kernels is
//' unavailable, the adaptive version \code{\link{adaFOHSIC}} should be
//' preferred.
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
//' The selection of the kernels with the forward algorithm implemented in
//' \code{\link{FOHSIC}} can be represented as a set of quadratic constraints.
//' This is owed to the quadratic form of the HSIC criterion. In this function,
//' we determine the matrices of the corresponding constraints. The output is a
//' list of matrices where the order is identical to the order of selection
//' of the kernels. The matrices are computed such the associated constraint is
//' nonnegative. For a length \eqn{n} of the list K, the total number of
//' constraints is \eqn{n - 1}.
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
