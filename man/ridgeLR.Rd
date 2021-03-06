% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/statistic.R
\name{ridgeLR}
\alias{ridgeLR}
\title{generates a closure for the computation of the likelihood ratio statistic
for the ridge prototype.}
\usage{
ridgeLR(K, mu = 0, sigma = 1, lambda = 1, tol = 1e-06, n_iter = 10000)
}
\arguments{
\item{K}{a single or a list of selected kernel similarity matrices.}

\item{mu}{mean of the response Y}

\item{sigma}{standard deviation of the response}

\item{lambda}{regularization parameter for the ridge prototype}

\item{tol}{convergence tolerance used a stopping criterion for the Newton-
Raphson algorithm}

\item{n_iter}{maximum number of iterations for the Newton-Raphson algorithm}
}
\value{
a closure for the calculation of the LR statistic for the ridge
prototype
}
\description{
The main inspiration for the kernel ridge prototype is the prototype concept
developed in Reid (2018, see references). A prototype is a synthetic scalar
variable that aggregates the effect of a set of variables in the outcome.
Here, we extend this concept to kernels, where the prototype is the
prediction of ridge regression with the selected kernels. In this function,
we implement a likelihood ratio (LR) statistic to test for the effect of the
the prototype on the outcome Y.
}
\details{
To maximize the likelihood objective function, we implement in the output
closure a Newton-Raphson algorithm that determines the maximum for each
input vector Y.

For our post-selection inference framework, The output closure is used to
compute the test statistics for both the replicates and the original outcome
in order to derive empirical \eqn{p}-values.
}
\examples{
n <- 30
p <- 20
K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
K <-  sapply(K, function(X) return(X \%*\% t(X) / dim(X)[2]), simplify = FALSE)
print(typeof(ridgeLR(K, mu = 0, sigma = 1, lambda = .1)) == "closure")

}
\references{
Reid, S., Taylor, J., & Tibshirani, R. (2018). A General
Framework for Estimation and Inference From Clusters of Features. Journal
of the American Statistical Association, 113(521), 280–293.
}
\seealso{
\code{\link{pcaLR}}

Other prototype: 
\code{\link{pcaLR}()}
}
\concept{prototype}
