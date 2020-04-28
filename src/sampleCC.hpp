// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat sampleCC(arma::field<arma::mat> A, NumericVector initial, int n_replicates,
                   double mu, double sigma, int n_iter, int burn_in);
