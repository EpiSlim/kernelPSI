#' implements the sequence kernel association test for GWAS data
#'
#' The SKAT test is a quadratic test of association between a
#' phenotype of interest and a genomic region. One of the main
#' benefits of the SKAT test is the incorporation of nonlinear
#' effects through the use of a kernel similarity matrix in the
#' quadratic form. For instance, the identical-by-state (IBS) kernel
#' which computes the number of identical alleles between two samples
#' can be used.
#'
#' The null hypothesis in the SKAT test is the absence of effects
#' of the SNPs within the region of interest and the outcome. Under the null,
#' the distribution of the test statistic is a weighted sum of chi-square
#' distributions whose quantiles are computed using the davies formula.
#'
#' @param K list of kernel similarity matrices. The sum kernel is used in
#' the quadratic form.
#' @param Y response vector
#' @param sigma standard deviation of the response Y
#'
#' @return \eqn{p}-value of the SKAT test
#'
#' @examples
#' n <- 30
#' p <- 20
#' K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
#' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
#' Y <- rnorm(n)
#' SKAT(Y, K)
#'
#' @references Wu, M. C., Lee, S., Cai, T., Li, Y., Boehnke, M., & Lin, X.
#' (2011). Rare-variant association testing for sequencing data with the
#' sequence kernel association test. American Journal of Human Genetics,
#' 89(1), 82–93.
#'
#' @export
SKAT <- function(Y, K, sigma = 1){
 if (is.list(K)) {
    Ksum <- Reduce(`+`, K)
  } else {
    Ksum <- K
 }

 Q <- drop(t(Y - mean(Y)) %*% Ksum %*% (Y - mean(Y)))
 Pmat <- sigma * (pracma::eye(dim(Ksum)[1]) - pracma::ones(dim(Ksum)[1]) / (dim(Ksum)[1]))
 eigD <- eigen(Pmat %*% Ksum %*% Pmat)[["values"]]
 pvalue <- CompQuadForm::davies(Q, eigD, sigma = 1, lim = 10000, acc = 0.0001)[["Qq"]]

 return(pvalue)
}
