#' generates a closure for the computation of the likelihood ratio statistic
#' for the ridge prototype.
#'
#' The main inspiration for the kernel ridge prototype is the prototype concept
#' developed in Reid (2018, see references). A prototype is a synthetic scalar
#' variable that aggregates the effect of a set of variables in the outcome.
#' Here, we extend this concept to kernels, where the prototype is the
#' prediction of ridge regression with the selected kernels. In this function,
#' we implement a likelihood ratio (LR) statistic to test for the effect of the
#' the prototype on the outcome Y.
#'
#' To maximize the likelihood objective function, we implement in the output
#' closure a Newton-Raphson algorithm that determines the maximum for each
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
#' @return a closure for the calculation of the LR statistic for the ridge
#' prototype
#'
#' @examples
#' n <- 30
#' p <- 20
#' K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
#' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
#' print(typeof(ridgeLR(K, mu = 0, sigma = 1, lambda = .1)) == "closure")
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
ridgeLR <- function(K, mu = 0, sigma = 1, lambda = 1, tol = 1e-6, n_iter = 1e+4) {
  if (is.list(K)) {
    Ksum <- Reduce(`+`, K)
  } else {
    Ksum <- K
  }

  decompK <- eigen(Ksum)
  eigsH <- decompK$values / (decompK$values + lambda)
  H <- decompK$vectors %*% diag(eigsH) %*% t(decompK$vectors)

  statistic <- function(Z) {
    theta <- 0.1
    gap <- tol + 1
    iter <- 0

    while (abs(gap) > tol && iter < n_iter) {
      gap <- (-sum(eigsH / (1 - theta * eigsH)) + drop(t(Z - mu) %*% H %*% Z) / sigma^2 -
             theta * drop(t(Z) %*% H %*% H %*% Z) / sigma^2) /
             (-sum(eigsH^2 / (1 - theta * eigsH)^2) - drop(t(Z) %*% H %*% H %*% Z) / (sigma^2))
      theta <- theta - gap
      iter <- iter + 1
    }

    R <- 2 * sum(log(1 - theta * eigsH)) + 2 * theta * drop(t(Z - mu) %*% H %*% Z) / sigma^2 -
         theta^2 * drop(t(Z) %*% H %*% H %*% Z) / sigma^2

    return(R)
  }

  return(statistic)
}

#' generates a closure for the computation of the likelihood ratio statistic
#' for the kernel PCA prototype.
#'
#' This function implements the same prototype statistics in the
#' \code{\link{ridgeLR}} function, but for kernel principal component regression
#' (see reference). In our simulations, we observed that this method
#' underperforms the ridge prototype. The main benefit of this approach is the
#' possibility of exact post-selection without the need for replicates sampling.
#'
#' @param K a single or a list of selected kernel similarity matrices.
#' @param mu marginal mean of the response Y
#' @param sigma standard deviation of the response
#'
#' @return a closure for the calculation of the LR statistic for the kernel
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

  if (all(class(Ksum) != "kernelMatrix")) {
    Ksum <- kernlab::as.kernelMatrix(Ksum, center = FALSE)
  }

  Krot <- kernlab::kpca(Ksum)@rotated
  H <- Krot %*% pracma::pinv(Krot)
  M <- sum(eigen(H, symmetric = TRUE, only.values = TRUE)[["values"]] > 0.5) # only possible eigenvalues: 0 or 1

  statistic <- function(Z) {
    roots <- Re(polyroot(c(- M + drop(t(Z - mu) %*% H %*% Z) / sigma^2, - drop(t(2 * Z - mu) %*% H %*% Z) / sigma^2,
                         drop(t(Z) %*% H %*% Z) / sigma^2)))
    roots <- roots[roots < 1, drop  = FALSE]
    R <- max(sapply(roots,
                        function(r) 2 * M * log(1 - r) + 2 * r * drop(t(Z - mu) %*% H %*% Z) / sigma^2 -
                        r^2 * drop(t(Z) %*% H %*% Z)/ sigma^2))

    return(R)
  }
}


#' computes a valid significance value for the effect of the selected kernels
#' on the outcome
#'
#' In this function, we compute an empirical \eqn{p}-value for the effect of a
#' subset of kernels on the outcome. A number of statistics are supported in
#' this function : ridge regression, kernel PCA and the HSIC criterion. The
#' \eqn{p}-values are determined by comparing the statistic of the original
#' response vector to those of the replicates. We use the \code{\link{sampleH}}
#' function to sample replicates of the response in the acceptance region of
#' the selection event.
#'
#' For valid inference on hundreds of samples, we recommend setting the number
#' of replicates to \eqn{50000} and the number of burn-in iterations to
#' \eqn{10000}. These ranges are to be increased for higher sample sizes.
#'
#' @param Y the response vector
#' @param K_select list of selected kernel
#' @param constraints list of quadratic matrices modeling the selection of the
#' kernels in \code{K_select}
#' @param method test statistic. Must be one of the following: \code{ridge} for
#' log-likelihood ratio for ridge regression, \code{pca} for log-likelihood for
#' kernel PCA, \code{hsic} for HSIC measures, or \code{all} to obtain
#' significance values for all three former methods.
#' @param mu mean of the response
#' @param sigma standard deviation of the response
#' @param lambda regularization parameter for ridge regression.
#' @param n_replicates number of replicates for the hit-and-run sampler in
#' \code{\link{sampleH}}
#' @param burn_in number of burn_in iteration in \code{\link{sampleH}}
#'
#' @return $p$-values for the chosen methods
#'
#' @examples
#' n <- 30
#' p <- 20
#' K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
#' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
#' Y <- rnorm(n)
#' L <- Y %*% t(Y)
#' selectK <- FOHSIC(K, L, mKernels = 2)
#' constraintFO <- forwardQ(K, selectK)
#' kernelPSI(Y, K[selectK], constraintFO, method = "ridge")
#'
#' @export
kernelPSI <- function(Y, K_select, constraints, method = "all",
                      mu = 0, sigma = 1, lambda = 1,
                      n_replicates = 5000, burn_in = 1000){

  Y <- drop(Y)

  samples <- sampleH(constraints, Y,
                     n_replicates = n_replicates, burn_in = burn_in,
                     mu = mu, sigma = sigma)

  pvalues <- list()

  if (any(method == "all") | any("ridge" %in% method)){
    newtonR <- ridgeLR(K_select, mu = mu, sigma = sigma, lambda = lambda)
    sampleR <- newtonR(Y)
    distR <- apply(samples, 2, newtonR)
    pvalues[["ridge"]] <- sum(distR > sampleR) / (dim(samples)[2])
  }

  if (any(method == "all") | any("pca" %in% method)){
    pcaR <- pcaLR(K_select, mu = 0)
    sampleR <- pcaR(Y)
    distR <- apply(samples, 2, pcaR)
    pvalues[["pca"]] <- sum(distR > sampleR) / (dim(samples)[2])
  }

  if (any(method == "all") | any("hsic" %in% method)){
    selectQQ <- quadHSIC(Reduce(`+`, K_select))
    sampleR <- drop(Y %*% selectQQ %*% Y)
    distR <- apply(samples, 2, function(s) return(s %*% selectQQ %*% s))
    pvalues[["hsic"]] <- sum(distR > sampleR) / (dim(samples)[2])
  }

  return(pvalues)

}
