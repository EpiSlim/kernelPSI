#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

//' samples within the acceptance region defined by the kernel selection event
//'
//' Similarly to the fixed variant, the adaptive selection of the 
//' kernels in a forward fashion can also be modeled with a set of 
//' quadratic constraints. The constraints for adaptive selection can be split
//' into two subsets. The first subset encodes the order of selection of the 
//' kernels, while the second subset encodes the selection of the number of the
//' kernels. The two subsets are equally sized (\code{length(K) - 1}) and are 
//' sequentially included in the output list. 
//' 
//' @param K list kernel similarity matrices
//' @param select integer vector containing the order of selection of the kernels 
//' in \code{K}. Typically, the \code{selection} field of the output of 
//' \code{\link{FOHSIC}}. 
//' @param n number of selected kernels. Typically, the \code{n} field of the 
//' output of \code{\link{adaFOHSIC}}. 
//'
//' @return list of matrices modeling the quadratic constraints of the
//' adaptive selection event
//' 
//' @references Loftus, J. R., & Taylor, J. E. (2015). Selective inference in 
//' regression models with groups of variables.
//'
//' @examples
//' n <- 50
//' p <- 20
//' K <- replicate(8, matrix(rnorm(n*p), nrow = n, ncol = p), simplify = FALSE)
//' K <-  sapply(K, function(X) return(X %*% t(X) / dim(X)[2]), simplify = FALSE)
//' L <- matrix(rnorm(n*p), nrow = n, ncol = p)
//' L <-  L %*% t(L) / p
//' adaS <- adaFOHSIC(K, L)
//' listQ <- adaQ(K, select = adaS[["selection"]], n = adaS[["n"]])
//' @export

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
