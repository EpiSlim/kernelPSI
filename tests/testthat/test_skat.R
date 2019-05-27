context("SKAT")

test_that("SKAT runs for a single or a list of similarity matrices", {
  n <- 50
  p <- 30
  K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
  K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
  Y <- rnorm(n)
  
  expect_type(SKAT(Y, K), "double")
  expect_silent(SKAT(Y, K[[1]]))

})