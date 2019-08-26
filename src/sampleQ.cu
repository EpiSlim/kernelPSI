#define ARMA_ALLOW_FAKE_GCC

#define VIENNACL_WITH_CUDA
#define VIENNACL_WITH_OPENMP
#define VIENNACL_WITH_ARMADILLO 1

#include <ctime>
#include <ratio>
#include <chrono>
using namespace std::chrono;

#include <RcppArmadillo.h>
// #include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;


// Cuda headers
#include <cuda.h>
#include <curand.h>

// Thrust headers
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <thrust/random/uniform_real_distribution.h>


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

template<class T>
struct normcdf_f{
    T mu;
    T sigma;
    normcdf_f(T _mu, T _sigma){
        mu=_mu;
        sigma=_sigma;
    }
    __host__ __device__ T operator()(T &x) const{
        return normcdf((x-mu)/sigma);
    }
};

template<class T>
struct normcdfinv_f{
    T mu;
    T sigma;
    normcdfinv_f(T _mu, T _sigma){
        mu=_mu;
        sigma=_sigma;
    }
    __host__ __device__ T operator()(T &x) const{
        return (normcdfinv(x) * sigma + mu);
    }
};

double sampleQ(arma::field<arma::mat> A, arma::vec initial, int n_replicates,
                  const double mu = 0.0, const double sigma = 1.0,
                  int n_iter = 1.0e+5, int burn_in = 1.0e+3)
{
    // Setting the generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // Functors
    normcdf_f<double> normcdf_functor(mu, sigma);
    normcdfinv_f<double> normcdfinv_functor(mu, sigma);

    // Initialization
    int n = initial.size();
    thrust::device_vector<double> candidateN(n), candidateQ(initial.begin(), initial.end());
    thrust::device_vector<double> qsamples(n*(n_replicates + burn_in), 0);
    thrust::device_vector<double> constraints(n*n*A.n_elem, 0);

    thrust::transform(candidateQ.begin(), candidateQ.end(), candidateN.begin(), normcdf_functor);

    arma::mat matA(n*A.n_elem, n);
    for (int r = 0; r < A.n_elem; ++r){
        matA(n*r, 0, size(A(r))) = A(r); // Regrouping the list of matrices in a single GPU matrix
    }
    thrust::copy(matA.begin(), matA.end(), constraints.begin());

    // sample a uniform vector
    double leftQ = 0, rightQ = 1;
    double lambda;

    /* Generate n floats on device */
    curandGenerateUniform(gen, &lambda, 1);

    // Rejection sampling
    thrust::device_vector<double> theta(n, 0);
    thrust::device_vector<double> boundA(n, 0), boundB(n, 0);

    return lambda;
}
