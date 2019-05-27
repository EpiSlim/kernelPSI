context("statistics")

test_that("ridgeLR and maxLR return a closure", {
  n <- 30
  p <- 20
  K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
  K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
  
  expect_type(ridgeLR(K, mu = 0, sigma = 1, lambda = .1), "closure")
  expect_type(pcaLR(K), "closure")
})

test_that("kernelPSI accepts all three statistics together", {
  n <- 30
  p <- 20
  K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
  K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
  Y <- rnorm(n)
  L <- Y %*% t(Y)
  selectK <- FOHSIC(K, L, mKernels = 2)
  constraintFO <- forwardQ(K, selectK)
  expect_length(
    kernelPSI(Y, K[selectK], constraintFO, method = c("ridge", "pca", "hsic")), 3
    )
})