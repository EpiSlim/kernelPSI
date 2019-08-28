#define ARMA_ALLOW_FAKE_GCC

#define VIENNACL_WITH_CUDA
#define VIENNACL_WITH_OPENMP
#define VIENNACL_WITH_ARMADILLO 1

#include <RcppArmadillo.h>
// #include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

// ViennaCL headers
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/forwards.h"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/sum.hpp"

#include "hsic.h"


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
double statC(arma::vec sample, arma::mat replicates, arma::field<arma::mat> K){

    // Compute the sum kernel
    int n = sample.size();
    arma::vec stat(replicates.n_cols);
    arma::mat Ksum(n, n, arma::fill::zeros);
    for (int r = 0; r < K.n_elem; ++r){
       Ksum += K(r);
    }

    // Transfer data to GPU
    viennacl::matrix<double> hsicCL(n, n);
    viennacl::matrix<double> replicatesCL(replicates.n_rows, replicates.n_cols), prodCL(replicates.n_rows, replicates.n_cols);
    viennacl::vector<double> sampleCL(n);

    copy(quadHSIC(Ksum), hsicCL);
    copy(replicates, replicatesCL);
    copy(sample, sampleCL);

    // Compute the statistic
    prodCL = viennacl::linalg::prod(hsicCL, replicatesCL);
    prodCL = viennacl::linalg::element_prod(prodCL, replicatesCL);
    viennacl::vector<double> statCL = viennacl::linalg::column_sum(prodCL);

    copy(statCL, stat);
    double statS = viennacl::linalg::inner_prod(viennacl::linalg::prod(hsicCL, sampleCL), sampleCL);

    double pvalue = arma::sum(statS > stat)/ (double) replicates.n_cols;

    return pvalue;
}
