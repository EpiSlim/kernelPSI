context("Rcpp")

test_that("adaFOHSIC generates the right output", {
  n <- 50
  p <- 20
  K <- replicate(5, matrix(rnorm(n * p), nrow = n, ncol = p), simplify = FALSE)
  L <- matrix(rnorm(n * p), nrow = n, ncol = p)
  K <- sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
  L <- L %*% t(L) / p
  adaS <- adaFOHSIC(K, L)

  expect_equal(names(adaS), c("selection", "n"))
  expect_length(adaS[["selection"]], length(K))
  expect_lte(adaS[["n"]], length(K))
  expect_equal(sort(adaS[["selection"]]), seq_along(K))
})

test_that("adaQ generates a list of the correct length", {
  n <- 50
  p <- 20
  n_kernels <- 10
  K <- replicate(n_kernels, matrix(rnorm(n * p), nrow = n, ncol = p), simplify = FALSE)
  K <- sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
  L <- matrix(rnorm(n * p), nrow = n, ncol = p)
  L <- L %*% t(L) / p
  adaS <- adaFOHSIC(K, L)

  expect_length(
    adaQ(K, select = adaS[["selection"]], n = adaS[["n"]]),
    2 * (n_kernels - 1)
  )
})

test_that("FOHSIC generates a valid list of indices", {
  n <- 50
  p <- 20
  n_kernels <- 5
  m_kernels <- 2
  K <- replicate(5, matrix(rnorm(n * p), nrow = n, ncol = p), simplify = FALSE)
  L <- matrix(rnorm(n * p), nrow = n, ncol = p)
  K <- sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
  L <- L %*% t(L) / p
  selection <- FOHSIC(K, L, m_kernels)

  expect_length(selection, m_kernels)
  expect_true(all(selection %in% seq(n_kernels)))
})


test_that("forwardQ generates a list of the correct length", {
  n <- 50
  p <- 20
  n_kernels <- 5
  K <- replicate(n_kernels, matrix(rnorm(n * p), nrow = n, ncol = p), simplify = FALSE)
  K <- sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)

  expect_length(forwardQ(K, select = c(4, 1)), n_kernels - 1)
})

test_that("HSIC and quadHSIC provide the same estimate", {
  n <- 50
  p <- 20
  K <- matrix(rnorm(n * p), nrow = n, ncol = p)
  K <- K %*% t(K) / p
  Y <- rnorm(n)
  L <- Y %*% t(Y)

  expect_equal(HSIC(K, L), drop(Y %*% quadHSIC(K) %*% Y) / (n * (n - 3)))
})

test_that("sampleH drawn samples are within the acceptance region", {
  n <- 30
  p <- 20
  n_replicates <- 50
  K <- replicate(5, matrix(rnorm(n * p), nrow = n, ncol = p), simplify = FALSE)
  K <- sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
  Y <- rnorm(n)
  L <- Y %*% t(Y)
  selection <- FOHSIC(K, L, 2)
  constraintQ <- forwardQ(K, select = selection)
  samples <- sampleH(
    A = constraintQ, initial = Y,
    n_replicates = n_replicates, burn_in = 20
  )

  expect_equal(dim(samples), c(n, n_replicates))
  expect_true(all(apply(
    samples, 2,
    function(s) return(sapply(constraintQ, function(q) return(s %*% q %*% s >= 0)))
  )))
})
