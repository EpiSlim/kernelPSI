#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat sampleH(arma::field<arma::mat> A, NumericVector initial, int n_replicates,
                  double mu = 0.0, double sigma = 1.0,
                  int n_iter = 1.0e+5, int burn_in = 1.0e+4)
{

    int n = initial.size();
    arma::mat qsamples(n, n_replicates + burn_in, arma::fill::zeros);
    arma::mat candidates(n, n_replicates + burn_in + 1, arma::fill::zeros);
    candidates.col(0) = Rcpp::as<arma::vec>(wrap(pnorm(initial, mu, sigma)));
    arma::vec candidateO, candidateN = Rcpp::as<arma::vec>(wrap(pnorm(initial, mu, sigma)));

    // Randomly sample in the sphere unit
    arma::mat theta(n, n_replicates + burn_in, arma::fill::randn);
    theta = normalise(theta, 2, 0);

    // Rejection sampling
    arma::vec cdt(A.n_elem);
    arma::vec::iterator l;
    arma::vec boundA, boundB;
    int r;
    for (int s = 0; s < (n_replicates + burn_in); ++s)
    {
        candidateO = candidateN;
        for (int iter = 0; iter < n_iter; ++iter)
        {
            if (iter == n_iter) stop("The quadratic constraints cannot be satisfied");
            boundA = -(candidateO/theta.col(s));
            boundB = (1 - candidateO)/theta.col(s);
            double leftQ = std::max(boundA.elem(arma::find(theta.col(s) > 0)).max(),
                                    boundB.elem(arma::find(theta.col(s) < 0)).max());
            double rightQ = std::min(boundA.elem(arma::find(theta.col(s) < 0)).min(),
                                     boundB.elem(arma::find(theta.col(s) > 0)).min());
            double lambda = runif(1, leftQ, rightQ)[0];
            candidateN = candidateO + lambda * theta.col(s);
            qsamples.col(s) = Rcpp::as<arma::vec>(wrap(qnorm(as<NumericVector>(wrap(candidateN)), mu, sigma)));
            for(l = cdt.begin(), r = 0; l != cdt.end(); ++l, ++r)
            {
                *l = arma::as_scalar(qsamples.col(s).t() * A(r) * qsamples.col(s));
            }
            if (all(cdt >= 0)) break;

        }
    }

    return qsamples.cols(burn_in, n_replicates + burn_in - 1);
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//

/*** R
require("MASS")
require("pracma")
require("kernlab")
require("tmg")
require("glmnet")
require("bindata")
require("dHSIC")
require("prototest")
require("CompQuadForm")
require("compiler")
require("KODAMA")

n_kernels <- 10
m_kernels <- 3
size_kernels <- 5
candidate_kernels <- 3
n <- 1000
rho <- 0.6

n_replicates <- 50000
burn_in <- 10000
n_permutations <- 10000

corr <- outer(seq_len(size_kernels), seq_len(size_kernels),
              function(i, j) return(rho^(abs(i-j))))

type <- 1
theta <- 0.1

X <- replicate(n_kernels,
               MASS::mvrnorm(n, mu = rep(0, size_kernels), Sigma = corr),
               simplify = FALSE)
K <- replicate(n_kernels, rbfdot(sigma = 1 / size_kernels))

Xmat <- do.call("cbind", X)
Kmat <- sapply(seq_len(n_kernels),
               function(i) {kMatrix <- kernelMatrix(K[[i]], X[[i]]); return(as.kernelMatrix(kMatrix, center = TRUE))},
               simplify = FALSE)
Ksum <- Reduce(`+`, Kmat[seq_len(m_kernels)])

decompK <- eigen(Ksum)

set.seed(100)
Y <- as.matrix(theta * decompK$values[1] * decompK$vectors[, 1] + rnorm(n), ncol = 1)
Lmat <- kernelMatrix(new("vanillakernel"), Y)

FOHSIC <- function(K, L, mKernels = 1) {

  # Initialization
  HSICcandidate <- sapply(K, function(k) return(HSIC(k, L)))

  selection <- which.max(HSICcandidate)
  if (mKernels == 1) return(selection)

  Ksum <- K[[selection]]
  for (m in seq(mKernels - 1)) {
    Kcandidate <- K[-selection]
    HSICcandidate <- sapply(Kcandidate, function(k) return(HSIC(Ksum + k, L)))
    idx <- seq_along(K)[-selection][which.max(HSICcandidate)]
    Ksum <- Ksum + K[[idx]]
    selection <- c(selection, idx)
  }

  return(selection)
}

HSIC <- function(K, L) {
  n <- dim(K)[1]
  K <- K - diag(diag(K))
  L <- L - diag(diag(L))
  KL <- K %*% L

  hsic <- (Trace(KL) + Reduce(`+`, K) * Reduce(`+`, L) / ((n - 1) * (n - 2)) -
    2 * Reduce(`+`, KL) / (n - 2)) / (n * (n - 3))

  return(hsic)
}

quadHSIC <- function(K) {
  n <- dim(K)[[1]]
  Q <- (K - diag(diag(K)) +
    Reduce(`+`, K - diag(diag(K))) * (ones(n) - eye(n)) / ((n - 1) * (n - 2)) -
    (2 / (n - 2)) * (ones(n) %*% K - diag(diag(ones(n) %*% K)) -
      ones(n) %*% diag(diag(K)) + diag(diag(ones(n) %*% diag(diag(K))))))

  return(Q)
}

forwardQ <- function(K, select) {
  constraintQ <- list()

  if (length(select) > 1) {
    for (s in seq(length(select) - 1)) {
      constraintQ[[s]] <- quadHSIC(K[[select[s]]]) - quadHSIC(K[[select[s + 1]]])
    }
  }

  constraintQ <- append(
    constraintQ,
    sapply(seq_along(K)[-select],
      function(s) return(
          quadHSIC(K[[tail(select, 1)]]) - quadHSIC(K[[s]])
        ),
      simplify = FALSE
    )
  )

  return(constraintQ)
}

selectFOHSIC <- FOHSIC(Kmat, Lmat, mKernels = candidate_kernels)
constraintFO <- forwardQ(Kmat, selectFOHSIC)
samples <- sampleH(constraintFO, Y,
                   n_replicates = n_replicates, burn_in = burn_in,
                   mu = 0, sigma = 1)

*/
