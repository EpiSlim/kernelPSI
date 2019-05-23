#' is a wrapper function for the Hamiltonian Monte-Carlo sampler
#'
#' In addition to our sampler \code{\link{sampleH}}, we include another sampler 
#' from the literature (see references) to sample replicates within the 
#' acceptance region. Similary to \code{\link{sampleH}}, \code{\link{sampleQ}}
#' covers second-order constraints for truncated gaussians and is based on 
#' Monte-Carlo chain iterative sampling. 
#' 
#' In our simulatons, \code{\link{sampleH}} was substantially faster 
#' \code{\link{sampleQ}}. 
#' 
#' @param Q list of matrices modeling the quadratic constraints of the
#' selection event
#' @param initial initialization sample. This sample must belong to the
#' acceptance region given by \code{Q}. In practice, this parameter is set
#' to the outcome of the original dataset.
#' @param n_replicates total number of replicates to be generated
#' @param mu mean of the outcome
#' @param sigma standard deviation of the outcome
#' @param n_iter maxmimum number of attempts for the sampling of an admissible 
#' point. Only used if initial is \code{NULL}. 
#' @param burn_in number of burn-in iterations
#'
#' @return a matrix with \code{n_replicates} columns where each column
#' contains a sample within the acceptance region
#'
#' @references Pakman, A., & Paninski, L. (2014). Exact Hamiltonian Monte Carlo
#' for truncated multivariate gaussians. Journal of Computational and Graphical 
#' Statistics, 23(2), 518–542.
#' 
#' @seealso \code{\link[tmg]{rtmg}}
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
#' samples <- sampleQ(Q = constraintQ, initial = Y,
#'                    n_replicates = 50, burn_in = 20)
#' @export
sampleQ <- function(Q, initial = NULL, n_replicates = 1e+3, mu = 0, sigma = 1, burn_in = 1e+2, n_iter = 1e+4){
    iter <- 1
    n <- dim(Q[[1]])[1]
    samples <- NULL

    if (.Platform$OS.type == 'windows') {
        sink("null")
    } else {
        sink("/dev/null")
    }
    
    if (is.null(initial)) {
      while (is.null(samples) && iter <= n_iter) {
        iter <- iter + 1
        samples <- tryCatch(
          tmg::rtmg(n = n_replicates, M = pracma::eye(n) / sigma^2, r = rep(mu, n) / sigma^2,
                    initial = stats::rnorm(n, mean = mu, sd = sigma),
                    f = NULL, g = NULL, burn.in = burn_in,
                    q = Q),
          error = function(e) return(NULL)
        )
      }
    } else {
      samples <- tmg::rtmg(n = n_replicates, M = pracma::eye(n) / sigma^2, r = rep(mu, n) / sigma^2,
                           initial = initial,
                           f = NULL, g = NULL, burn.in = burn_in,
                           q = Q)
    }

    sink()

    return(samples)
}
