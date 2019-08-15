#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

//' samples within the acceptance region defined by the kernel selection event
//'
//' To approximate the distribution of the test statistics, we iteratively
//' sample replicates of the response in order to generate replicates
//' of the test statistics. The response replicates are iteratively sampled
//' within the acceptance region of the selection event. The goal of the
//' constrained sampling is to obtain a valid post-selection distribution of
//' the test statistic. To perform the constrained sampling, we develop a hit-and-run
//' sampler based on the hypersphere directions algorithm (see references).
//'
//' Given the iterative nature of the sampler, a large number of
//' \code{n_replicates} and \code{burn_in} iterations is needed to correctly
//' approximate the test statistics distributions.
//'
//' For high-dimensional responses, and depending on the initialization, the
//' sampler may not scale well to generate tens of thousands of replicates
//' because of an intermediate rejection sampling step.
//'
//' @param A list of matrices modeling the quadratic constraints of the
//' selection event
//' @param initial initialization sample. This sample must belong to the
//' acceptance region given by \code{A}. In practice, this parameter is set
//' to the outcome of the original dataset.
//' @param n_replicates total number of replicates to be generated
//' @param mu mean of the outcome
//' @param sigma standard deviation of the outcome
//' @param n_iter maximum number of rejections for the parameter \eqn{\lambda}
//' in a single iteration
//' @param burn_in number of burn-in iterations
//'
//' @return a matrix with \code{n_replicates} columns where each column
//' contains a sample within the acceptance region
//'
//' @references Berbee, H. C. P., Boender, C. G. E., Rinnooy Ran, A. H. G.,
//' Scheffer, C. L., Smith, R. L., & Telgen, J. (1987). Hit-and-run algorithms
//' for the identification of non-redundant linear inequalities. Mathematical
//' Programming, 37(2), 184–207.
//'
//' @references Belisle, C. J. P., Romeijn, H. E., & Smith, R. L. (2016).
//' HIT-AND-RUN ALGORITHMS FOR GENERATING MULTIVARIATE DISTRIBUTIONS,
//' 18(2), 255–266.
//'
//' @examples
//' n <- 30
//' p <- 20
//' K <- replicate(5, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
//' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
//' Y <- rnorm(n)
//' L <- Y %*% t(Y)
//' selection <- FOHSIC(K, L, 2)
//' constraintQ <- forwardQ(K, select = selection)
//' samples <- sampleH(A = constraintQ, initial = Y,
//'                    n_replicates = 50, burn_in = 20)
//' @export
// [[Rcpp::export]]
arma::mat sampleH(arma::field<arma::mat> A, NumericVector initial, int n_replicates,
                  double mu = 0.0, double sigma = 1.0,
                  int n_iter = 1.0e+5, int burn_in = 1.0e+3)
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
