context("Likelihood")

test_that("maxLR and anovaLR generate values comprised between 0 and 1", {
  n <- 50
  p <- 30
  X <- matrix(rnorm(n*p), ncol = p, nrow = n)
  Y <- rnorm(n)
  p_max <- maxLR(X, Y)
  p_anova <- anovaLR(X, Y)
  
  expect_lte(p_max, 1)
  expect_gte(p_max, 0)
  
  expect_lte(p_anova, 1)
  expect_gte(p_anova, 0)
})

test_that("anovaLR throws an error when p > n, as opposed to maxLR", {
  n <- 50
  p <- 100
  X <- matrix(rnorm(n*p), ncol = p, nrow = n)
  Y <- rnorm(n)
  
  expect_silent(maxLR(X, Y))
  expect_error(anovaLR(X, Y))
})