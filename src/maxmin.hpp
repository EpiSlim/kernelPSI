// CUDA headers
#include "cublas_v2.h"
#include "curand.h"
#include <cuda.h>
#include <cuda_runtime.h>

// [[Rcpp::plugins(cpp11)]]
template <int SIGN, int elem_per_thread = 2, int thread_per_block = 256>
__host__ double cuda_find_max(const double *d_data, const int data_len);
