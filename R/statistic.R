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
