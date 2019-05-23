#' implements the sequence kernel association
#'
#' The maximum likelihood ratio test is a classical goodness-of-fit
#' test for linear models. Mathematically speaking, the average
#' residual sum of squares for an ordinary least squares (OLS) is
#' approximated as a chi-square distribution to generate a \eqn{p}-value.
#'
#' null hypothesis
#'
#'
#' @param K list of kernel similarity matrices
#' @param Y response vector
#' @param sigma standard deviation of the response Y
#'
#' @return \eqn{p}-value of the SKAt test
#'
#' @examples
#' n <- 50
#' p <- 20
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#' Y <- rnorm(n)
#' stat.likelihood <- maxLR(X, Y)
#'
#' @export
SKAT <- function(Y, K, sigma = 1){
 if (is.list(K)) {
    Ksum <- Reduce(`+`, K)
  } else {
    Ksum <- K
 }

 Q <- drop(t(Y - mean(Y)) %*% Ksum %*% (Y - mean(Y)))
 Pmat <- sigma * (pracma::eye(dim(Ksum)[1]) - pracma::ones(dim(Ksum)[1]) / (dim(Ksum)[1]))
 eigD <- eigen(Pmat %*% Ksum %*% Pmat)[["values"]]
 pvalue <- CompQuadForm::davies(Q, eigD, sigma = 1, lim = 10000, acc = 0.0001)[["Qq"]]

 return(pvalue)
}
