#include <RcppArmadillo.h>
#include "hsic.h"
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

//' adaptively selects a subset of kernels in a forward fashion.
//'
//' This function is similar to the \code{\link{FOHSIC}} function. The only
//' difference lies in the adaptive selection of the number of causal kernels.
//' First, similarly to \code{\link{FOHSIC}}, the order of selection of the
//' \eqn{n} kernels in \code{K} is determined, and then, the size of the subset
//' of ordered kernels is chosen. The size is chosen as to maximize the overall
//' association with the kernel L.
//'
//' @param K list of kernel similarity matrices
//' @param L kernel similarity matrix for the outcome
//'
//' @return a list where the the first item \code{selection} is the order of
//' selection of all kernels in the list \code{K} and the second item is the
//' number of selected kernels.
//'
//' @examples
//' n <- 50
//' p <- 20
//' K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
//' L <- matrix(rnorm(n*p), nrow = n, ncol = p)
//' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
//' L <-  L %*% t(L) / p
//' adaS <- adaFOHSIC(K, L)
//' print(names(adaS) == c("selection", "n"))
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

//' models the forward selection of the kernels for the adaptive variant
//'
//' Similarly to the fixed variant, the adaptive selection of the
//' kernels in a forward fashion can also be modeled with a set of
//' quadratic constraints. The constraints for adaptive selection can be split
//' into two subsets. The first subset encodes the order of selection of the
//' kernels, while the second subset encodes the selection of the number of the
//' kernels. The two subsets are equally sized (\code{length(K) - 1}) and are
//' sequentially included in the output list.
//'
//' @param K list kernel similarity matrices
//' @param select integer vector containing the order of selection of the kernels
//' in \code{K}. Typically, the \code{selection} field of the output of
//' \code{\link{FOHSIC}}.
//' @param n number of selected kernels. Typically, the \code{n} field of the
//' output of \code{\link{adaFOHSIC}}.
//'
//' @return list of matrices modeling the quadratic constraints of the
//' adaptive selection event
//'
//' @references Loftus, J. R., & Taylor, J. E. (2015). Selective inference in
//' regression models with groups of variables.
//'
//' @examples
//' n <- 50
//' p <- 20
//' K <- replicate(8, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
//' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
//' L <- matrix(rnorm(n*p), nrow = n, ncol = p)
//' L <-  L %*% t(L) / p
//' adaS <- adaFOHSIC(K, L)
//' listQ <- adaQ(K, select = adaS[["selection"]], n = adaS[["n"]])
//' @export
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
