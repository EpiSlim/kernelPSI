# Maximum likelihood ratio test (RSS difference)  -----------------
maxLR <- function(X, Y) {
  glm.fit <- stats::glm(Y ~ X)
  ratio <- lmtest::lrtest(glm.fit)[2, "Pr(>Chisq)"]

  return(ratio)
}

# Anova testing (scaled RSS testing) -----------------
anovaLR <- function(X, Y) {
  glm.fit <- stats::glm(Y ~ X)
  ratio <- stats::anova(glm.fit, test = "LRT")[2, "Pr(>Chi)"]

  return(ratio)
}
