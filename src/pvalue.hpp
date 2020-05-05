// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

//' computes a p-value for the quadratic kernel association score
//' 
//' Computing exact \eqn{p}-values for quadratic association scores is often out of 
//' reach. Sampling replicates of the original outcome \code{sample} can help in this regard. 
//' The replicates are sampled using the function \code{\link{sampleH}}. Here, we
//' compute the statistics of these replicates and compare them to the statistic of the 
//' original outcome to determine an empirical \eqn{p}-value. 
//'
//' @param sample original outcome vector
//' @param replicates matrix of valid replicates. Each column of this matrix corresponds to a
//' distinct replicate
//' @param K a list of matrices corresponding to the Gram matrices of the selected kernels
//'
//' @return an empirical \eqn{p}-value
//' 
//' @examples
//' n <- 30
//' p <- 20
//' K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
//' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
//' Y <- rnorm(n)
//' L <- Y %*% t(Y)
//' selection <- FOHSIC(K, L, 2)
//' constraintQ <- forwardQ(K, select = selection)
//' samples <- sampleH(A = constraintQ, initial = Y,
//'                    n_replicates = 50, burn_in = 20)
//' pvalue <- statC(Y, samples, K[selection])
//' 
//' @export
// [[Rcpp::export]]
double pvalue(arma::vec sample, arma::mat replicates, arma::field<arma::mat> K);
