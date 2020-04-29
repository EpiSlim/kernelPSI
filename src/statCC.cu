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

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

__global__ 
void cuda_element_prod(int n, double *x, double *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    x[i] *= y[i];
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

double statCC(arma::vec sample, arma::mat replicates, arma::field<arma::mat> K){

    // Compute the sum kernel
    int n = sample.size();
    arma::vec stat(replicates.n_cols);
    arma::mat Ksum(n, n, arma::fill::zeros);
    for (int r = 0; r < K.n_elem; ++r){
       Ksum += K(r);
    }
    Ksum = quadHSIC(Ksum);


    // CUDA section
    cublasHandle_t handle;
    cublasStatus_t statHandle = cublasCreate( &handle );

    double *hsicCUDA, *replicatesCUDA, *prodCUDA, *sampleCUDA, *tmpCUDA, *statCUDA, *statSS;


    // Allocate all our host-side (CPU) and device-side (GPU) data
    cudaMallocManaged(&hsicCUDA, n * n * sizeof( double ));
    cudaMallocManaged(&replicatesCUDA, replicates.n_rows * replicates.n_cols * sizeof( double ));
    cudaMallocManaged(&prodCUDA, replicates.n_rows * replicates.n_cols * sizeof( double ));
    cudaMallocManaged(&sampleCUDA, n * sizeof( double ));
    cudaMallocManaged(&tmpCUDA, n * sizeof( double ));
    cudaMallocManaged(&statSS, sizeof( double ));

    // Copy data to CUDA objects
    cudaMemcpy(hsicCUDA, Ksum.memptr(), n * n * sizeof( double ), cudaMemcpyHostToDevice);
    cudaMemcpy(replicatesCUDA, replicates.memptr(), replicates.n_rows * replicates.n_cols * sizeof( double ), 
                cudaMemcpyHostToDevice);
    cudaMemcpy(sampleCUDA, sample.memptr(), n * sizeof( double ), cudaMemcpyHostToDevice); 

    // Set these constants so we get a simple matrix multiply with cublasDgemm
    double alpha = 1.0;
    double beta  = 0.0;

    // Computing the statistic for replicates
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, replicates.n_cols, n,
        &alpha,
        hsicCUDA, n,
        replicatesCUDA, replicates.n_rows,
        &beta,
        prodCUDA, n);

    int blockSize = 256;
    int numBlocks = (replicates.n_rows * replicates.n_cols + blockSize - 1) / blockSize;
    cuda_element_prod<<<numBlocks, blockSize>>>(replicates.n_rows * replicates.n_cols, prodCUDA, replicatesCUDA);

    // Computing statistic for original sample
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, 1, n,
        &alpha,
        hsicCUDA, n,
        sampleCUDA, n,
        &beta,
        tmpCUDA, n)

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        1, 1, n,
        &alpha,
        tmpCUDA, 1,
        sampleCUDA, 1,
        &beta,
        statSS, 1)
    
    cudaDeviceSynchronize();
    
    // Free resources
    cublasDestroy( handle );

    cudaFree( hsicCUDA );
    cudaFree( replicatesCUDA );
    cudaFree( prodCUDA );
    cudaFree( sampleCUDA );


    // Transfer data to GPU
    viennacl::matrix<double> hsicCL(n, n);
    viennacl::matrix<double> replicatesCL(replicates.n_rows, replicates.n_cols), prodCL(replicates.n_rows, replicates.n_cols);
    viennacl::vector<double> sampleCL(n);

    copy(Ksum, hsicCL);
    copy(replicates, replicatesCL);
    copy(sample, sampleCL);

    // Compute the statistic
    prodCL = viennacl::linalg::prod(hsicCL, replicatesCL);
    prodCL = viennacl::linalg::element_prod(prodCL, replicatesCL);
    viennacl::vector<double> statCL = viennacl::linalg::column_sum(prodCL);

    copy(statCL, stat);
    double statS = viennacl::linalg::inner_prod(viennacl::linalg::prod(hsicCL, sampleCL), sampleCL);

    double pvalue = arma::sum(stat > statS)/ (double) replicates.n_cols;

    return pvalue;
}
