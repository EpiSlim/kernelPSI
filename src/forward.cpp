#include <RcppArmadillo.h>
#include "hsic.h"
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

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
