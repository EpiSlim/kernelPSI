#include <RcppArmadillo.h>
//#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

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
