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
arma::mat sampleH(arma::field<arma::mat> A, arma::field<arma::vec> b, arma::vec c,
                  NumericVector initial, int n_replicates,
                  double mu = 0.0, double sigma = 1.0,
                  int n_iter = 1.0e+5, int burn_in = 50)
{

    int n = initial.size();
    arma::mat qsamples(n, n_replicates + burn_in, arma::fill::zeros);
    arma::mat candidates(n, n_replicates + burn_in + 1, arma::fill::zeros);
    candidates.col(0) = Rcpp::as<arma::vec>(wrap(pnorm(initial, mu, sigma)));

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
        for (int iter = 0; iter < n_iter; ++iter)
        {
            if (iter == n_iter) stop("The quadratic constraints cannot be satisfied");
            boundA = -(candidates.col(s)/theta.col(s));
            boundB = (1 - candidates.col(s))/theta.col(s);
            double leftQ = std::max(boundA.elem(arma::find(theta.col(s) > 0)).max(),
                                    boundB.elem(arma::find(theta.col(s) < 0)).max());
            double rightQ = std::min(boundA.elem(arma::find(theta.col(s) < 0)).min(),
                                     boundB.elem(arma::find(theta.col(s) > 0)).min());
            double lambda = runif(1, leftQ, rightQ)[0];
            candidates.col(s+1) = candidates.col(s) + lambda * theta.col(s);
            qsamples.col(s) = Rcpp::as<arma::vec>(wrap(qnorm(as<NumericVector>(wrap(candidates.col(s+1))), mu, sigma)));
            for(l = cdt.begin(), r = 0; l != cdt.end(); ++l, ++r)
            {
                *l = arma::as_scalar(qsamples.col(s).t() * A(r) * qsamples.col(s)) +
                     arma::dot(b(r), qsamples.col(s)) + c(r);
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
n <- 40
nconstraints <- 8
A <- replicate(nconstraints, matrix(rnorm(n*n), ncol = n, nrow = n), simplify = FALSE)
b <- replicate(nconstraints, rep(0, n), simplify = FALSE)
c <- rep(0, nconstraints)
initial = rep(0, n)
n_replicates = 50
test <- sampleH(A, b, c, initial, n_replicates)
*/
