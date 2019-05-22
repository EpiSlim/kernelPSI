#include <RcppArmadillo.h>
#include "hsic.h"
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

//' selects a fixed number of kernels which are most associated with the
//' outcome kernel.
//'
//' This function implements a foward algorithm for kernel selection. In the
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
//' selection <- adaFOHSIC(K, L)
//'
//' @export
// [[Rcpp::export]]
List adaFOHSIC(arma::field<arma::mat> K, arma::mat L)
{

    arma::vec hvec(K.n_elem);
    arma::vec::iterator h;
    int r;

    for (h = hvec.begin(), r = 0; h != hvec.end(); ++h, ++r)
    {
        *h = HSIC(K(r), L);
    }

    arma::uvec selection = sort_index(hvec, "descend");
    arma::vec HSICsum(K.n_elem);
    arma::mat Ksum(size(K(0)), arma::fill::zeros);
    for (h = HSICsum.begin(), r = 0; h != HSICsum.end(); ++h, ++r)
    {
        Ksum += K(selection(r));
        *h = HSIC(Ksum, L);
    }

    return List::create(Rcpp::Named("selection") = selection + 1,
                        Rcpp::Named("n") = index_max(HSICsum) + 1);

}

// [[Rcpp::export]]
arma::field<arma::mat> adaQ(arma::field<arma::mat> K, IntegerVector select, int n)
{
    arma::field<arma::mat> constraintQ(2 * (K.n_elem - 1));
    int s;
    for (s = 0; s != (select.size() - 1); ++s)
    {
        constraintQ(s) = quadHSIC(K(select(s) - 1)) - quadHSIC(K(select(s + 1) - 1));
    }

    arma::field<arma::mat> HSICsum(K.n_elem);
    arma::mat HSICroll(size(K(0)), arma::fill::zeros);
    for (s = 0; s != select.size(); ++s)
    {
        HSICroll += quadHSIC(K(select(s) - 1));
        HSICsum(s) = HSICroll;
    }

    int r = select.size() - 1;
    for(s = 0; s!= select.size(); ++s)
    {
        if (s != (n - 1))
        {
            constraintQ(r) = HSICsum(n - 1) - HSICsum(s);
            r += 1;
        }
    }

    return constraintQ;
}
