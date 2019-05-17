#ifndef HSIC_HEADER
#define HSIC_HEADER

#include <RcppArmadillo.h>

double HSIC(arma::mat K, arma::mat L);
arma::mat quadHSIC(arma::mat K);

#endif
