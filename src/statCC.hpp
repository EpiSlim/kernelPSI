// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double statCC(arma::vec sample, arma::mat replicates, arma::field<arma::mat> K);
