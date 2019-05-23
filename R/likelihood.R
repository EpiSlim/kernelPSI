#' implements the maximum likelihood ratio test
#'
#' The maximum likelihood ratio test is a classical goodness-of-fit
#' test for linear models. Mathematically speaking, the average
#' residual sum of squares for an ordinary least squares (OLS) is
#' approximated as a chi-square distribution to generate a \eqn{p}-value.
#'
#' The test is valid when the number of samples is larger than the number
#' of covariates. 
#'
#' @param X covariate matrix
#' @param Y response vector
#'
#' @return \eqn{p}-value of the test
#'
#' @family LR test
#'
#' @examples
#' n <- 50
#' p <- 20
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#' Y <- rnorm(n)
#' stat.likelihood <- maxLR(X, Y)
#'
#' @export
maxLR <- function(X, Y) {
  glm.fit <- stats::glm(Y ~ X)
  ratio <- lmtest::lrtest(glm.fit)[2, "Pr(>Chisq)"]

  return(ratio)
}

#' implements a scaled variant of the maximum likelihood ratio test
#'
#' Compared to \code{\link{maxLR}}, the residual sum of squares (RSS) is scaled
#' by the degrees of freedom of the model \eqn{df = n - k}, where \eqn{n} is
#' the number of samples and \eqn{k} is the number of covariates. In
#' \code{\link{maxLR}}, the RSS is instead averaged over \eqn{n}. Both estimators
#' are asymptotically equivalent, with minor differences for finite samples.
#' Further details in this \href{https://stats.stackexchange.com/a/155614}{link}.
#'
#' @param X covariate matrix
#' @param Y response vector
#'
#' @return \eqn{p}-value of the test
#'
#' @family LR test
#'
#' @examples
#' n <- 50
#' p <- 20
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#' Y <- rnorm(n)
#' stat.anova <- anovaLR(X, Y)
#'
#' @export
anovaLR <- function(X, Y) {
  glm.fit <- stats::glm(Y ~ X)
  ratio <- stats::anova(glm.fit, test = "LRT")[2, "Pr(>Chi)"]

  return(ratio)
}
