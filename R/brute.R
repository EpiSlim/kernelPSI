#' samples within the acceptance region defined by the kernel selection event
#'
#' To approximate the distribution of the test statistics, we iteratively
#' sample replicates of the response in order to generate replicates
#' of the test statistics. The response replicates are iteratively sampled
#' within the acceptance region of the selection event. The goal of the
#' constrained sampling is to obtain a valid post-selection distribution of
#' the test statistic. To perform the constrained sampling, we develop a hit-and-run
#' sampler based on the hypersphere directions algorithm (see references).
#'
#' Given the iterative nature of the sampler, a large number of
#' \code{n_replicates} and \code{burn_in} iterations is needed to correctly
#' approximate the test statistics distributions.
#'
#' For high-dimensional responses, and depending on the initialization, the
#' sampler may not scale well to generate tens of thousands of replicates
#' because of an intermediate rejection sampling step.
#'
#' @param A list of matrices modeling the quadratic constraints of the
#' selection event
#' @param mu mean of the outcome
#' @param sigma standard deviation of the outcome
#' @param n_iter maxmimum number of rejections for the parameter \eqn{\lambda}
#' in a single iteration
#' 
#' @return a matrix with \code{n_replicates} columns where each column
#' contains a sample within the acceptance region
#'
#' @examples
#' n <- 30
#' p <- 20
#' K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
#' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
#' Y <- rnorm(n) 
#' L <- Y %*% t(Y)  
#' selection <- FOHSIC(K, L, 2)
#' constraintQ <- forwardQ(K, select = selection)
#' samples <- sampleH(A = constraintQ, initial = Y,
#'                    n_replicates = 50, burn_in = 20)
#' @export
sampleB <- function(Q, mu = 0, sigma = 1, n_iter = 1e+4){
  n <- dim(Q[[1]])[1]
  cdt <- FALSE
  iter <- 1

  while (!cdt) {
    if (iter == n_iter) stop("The constraint can not be satisfied")
    iter <- iter + 1
    Y <- stats::rnorm(n, mean = mu, sd = sigma)
    cdt <- all(sapply(Q, function(q) return(Y %*% q[[1]] %*% Y)))
  }

  return(Y)

}
