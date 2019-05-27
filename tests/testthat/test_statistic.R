context("statistics")

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