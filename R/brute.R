#' @export
sampleB <- function(Q, mu = 0, sigma = 1, n_iter = 1e+4){
  n <- dim(Q[[1]][[1]])[1]
  cdt <- FALSE
  iter <- 1

  while (!cdt) {
    if (iter == n_iter) stop("The constraint can not be satisfied")
    iter <- iter + 1
    Y <- rnorm(n, mean = mu, sd = sigma)
    cdt <- all(sapply(Q, function(q) return(Y %*% q[[1]] %*% Y +  Y %*% q[[2]] + q[[3]] >= 0)))
  }

  return(Y)

}
