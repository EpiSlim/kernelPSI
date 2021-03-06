% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{FOHSIC}
\alias{FOHSIC}
\title{selects a fixed number of kernels which are most associated with the
outcome kernel.}
\usage{
FOHSIC(K, L, mKernels = 1L)
}
\arguments{
\item{K}{list of kernel similarity matrices}

\item{L}{kernel similarity matrix for the outcome}

\item{mKernels}{number of kernels to be selected}
}
\value{
an integer vector containing the indices of the selected kernels
}
\description{
This function implements a forward algorithm for kernel selection. In the
first step, the kernel which maximizes the HSIC measure with the outcome
kernel \code{L} is selected. In the subsequent iterations, the kernel which,
combined with the selected kernels maximizes the HSIC measure is selected.
For the sum kernel combination rule, the forward algorithm can be
simplified. The kernels which maximize the HSIC measure with the kernel
\code{L} are selected in a descending order.
}
\details{
\code{\link{FOHSIC}} implements the forward algorithm with a predetermined
number of kernels \code{mKernels}. If the exact number of causal kernels is
unavailable, the adaptive version \code{\link{adaFOHSIC}} should be
preferred.
}
\examples{
n <- 50
p <- 20
K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
L <- matrix(rnorm(n*p), nrow = n, ncol = p)
K <-  sapply(K, function(X) return(X \%*\% t(X) / dim(X)[2]), simplify = FALSE)
L <-  L \%*\% t(L) / p
selection <- FOHSIC(K, L, 2)

}
