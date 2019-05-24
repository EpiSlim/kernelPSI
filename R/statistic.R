#' generates a closure for the computation of the likelihood ratio statistic
#' for the ridge prototype.
#'
#' The main inspiration for the kernel ridge prototype is the prototype concept
#' developed in Reid (2018, see references). A prototype is a synethic scalar
#' variable that aggregates the effect of a set of variables in the outcome.
#' Here, we extend this concept to kernels, where the prototype is the
#' prediction of ridge regression with the selected kernels. In this function,
#' we implement a likelihood ratio (LR) statistic to test for the effect of the
#' the prototype on the outcome Y.
#'
#' To maximize the likelihood objective function, we implement in the output
#' closure a Newton-Raphson algorithm that determines the maxmimum for each
#' input vector Y.
#'
#' For our post-selection inference framework, The output closure is used to
#' compute the test statistics for both the replicates and the original outcome
#' in order to derive empirical \eqn{p}-values.
#'
#' @param K a single or a list of selected kernel similarity matrices.
#' @param mu mean of the response Y
#' @param sigma standard deviation of the response
#' @param lambda regularization parameter for the ridge prototype
#' @param tol convergence tolerance used a stopping criterion for the Newton-
#' Raphson algorithm
#' @param n_iter maximum number of iterations for the Newton-Raphson algorithm
#'
#' @return a closure for the calcuation of the LR statistic for the ridge
#' prototype
#'
#' @examples
#' n <- 30
#' p <- 20
#' K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
#' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
#' print(typeof(LR(K, mu = 0, sigma = 1, lambda = .1)) == "closure")
#'
#' @references Reid, S., Taylor, J., & Tibshirani, R. (2018). A General
#' Framework for Estimation and Inference From Clusters of Features. Journal
#' of the American Statistical Association, 113(521), 280–293.
#'
#' @family prototype
#'
#' @seealso \code{\link{pcaLR}}
#'
#' @export
LR <- function(K, mu = 0, sigma = 1, lambda = 1, tol = 1e-6, n_iter = 1e+4) {
  if (is.list(K)) {
    Ksum <- Reduce(`+`, K)
  } else {
    Ksum <- K
  }

  decompK <- eigen(Ksum)
  eigsH <- decompK$values / (decompK$values + lambda)
  H <- decompK$vectors %*% diag(eigsH) %*% t(decompK$vectors)

  statistic <- function(Y) {
    theta <- 0.1
    gap <- tol + 1
    iter <- 0

    while (abs(gap) > tol && iter < n_iter) {
      gap <- (-sum(eigsH / (1 - theta * eigsH)) + drop(t(Y - mu) %*% H %*% Y) / sigma^2 -
             theta * drop(t(Y) %*% H %*% H %*% Y) / sigma^2) /
             (-sum(eigsH^2 / (1 - theta * eigsH)^2) - drop(t(Y) %*% H %*% H %*% Y) / (sigma^2))
      theta <- theta - gap
      iter <- iter + 1
    }

    R <- 2 * sum(log(1 - theta * eigsH)) + 2 * theta * drop(t(Y - mu) %*% H %*% Y) / sigma^2 -
         theta^2 * drop(t(Y) %*% H %*% H %*% Y) / sigma^2

    return(R)
  }

  return(statistic)
}

#' generates a closure for the computation of the likelihood ratio statistic
#' for the kernel PCA prototype.
#'
#' This function implements the same prototype statistics in the
#' \code{\link{LR}} function, but for kernel principal component regression
#' (see reference). In our simulations, we observed that this method
#' underperforms the ridge prototype. The main benefit of this approach is the
#' possibility of exact post-selection without the need for replicates sampling.
#'
#' @param K a single or a list of selected kernel similarity matrices.
#' @param mu marginal mean of the response Y
#' @param sigma standard deviation of the response
#'
#' @return a closure for the calcuation of the LR statistic for the kernel
#' PCA prototype
#'
#' @examples
#' n <- 30
#' p <- 20
#' K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
#' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
#' print(typeof(pcaLR(K, mu = 0, sigma = 1)) == "closure")
#'
#' @references Rosipal, R., Girolami, M., Trejo, L. J., & Cichocki, A. (2001).
#' Kernel PCA for feature extraction and de-noising in nonlinear regression.
#' Neural Computing and Applications, 10(3), 231–243.
#'
#' @family prototype
#'
#' @export
pcaLR <- function(K, mu = 0, sigma = 1) {
  if (is.list(K)) {
    Ksum <- Reduce(`+`, K)
  } else {
    Ksum <- K
  }

  if (class(Ksum) != "kernelMatrix") {
    Ksum <- kernlab::as.kernelMatrix(Ksum, center = FALSE)
  }

  Krot <- kernlab::kpca(Ksum)@rotated
  H <- Krot %*% pracma::pinv(Krot)
  M <- sum(eigen(H, symmetric = TRUE, only.values = TRUE)[["values"]] > 0.5) # only possible eigenvalues: 0 or 1

  statistic <- function(Y) {
    roots <- Re(polyroot(c(- M + drop(t(Y - mu) %*% H %*% Y) / sigma^2, - drop(t(2 * Y - mu) %*% H %*% Y) / sigma^2,
                         drop(t(Y) %*% H %*% Y) / sigma^2)))
    roots <- roots[roots < 1, drop  = FALSE]
    R <- max(sapply(roots,
                        function(r) 2 * M * log(1 - r) + 2 * r * drop(t(Y - mu) %*% H %*% Y) / sigma^2 -
                        r^2 * drop(t(Y) %*% H %*% Y)/ sigma^2))

    return(R)
  }
}
