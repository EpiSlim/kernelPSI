#include <RcppArmadillo.h>
#include "hsic.h"
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

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
