% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{adaFOHSIC}
\alias{adaFOHSIC}
\title{adaptively selects a subset of kernels in a forward fashion.}
\usage{
adaFOHSIC(K, L)
}
\arguments{
\item{K}{list of kernel similarity matrices}

\item{L}{kernel similarity matrix for the outcome}
}
\value{
a list where the the first item \code{selection} is the order of
selection of all kernels in the list \code{K} and the second item is the
number of selected kernels.
}
\description{
This function is similar to the \code{\link{FOHSIC}} function. The only
difference lies in the adaptive selection of the number of causal kernels.
First, similarly to \code{\link{FOHSIC}}, the order of selection of the
\eqn{n} kernels in \code{K} is determined, and then, the size of the subset
of ordered kernels is chosen. The size is chosen as to maximize the overall
association with the kernel L.
}
\examples{
n <- 50
p <- 20
K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
L <- matrix(rnorm(n*p), nrow = n, ncol = p)
K <-  sapply(K, function(X) return(X \%*\% t(X) / dim(X)[2]), simplify = FALSE)
L <-  L \%*\% t(L) / p
adaS <- adaFOHSIC(K, L)
print(names(adaS) == c("selection", "n"))

}
