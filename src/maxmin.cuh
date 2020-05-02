// CUDA headers
#include "cublas_v2.h"
#include "curand.h"
#include <cuda.h>
#include <cuda_runtime.h>

template <int SIGN, int BLKSZ>
__global__ void cuda_max_reduce(const double *d_data, const int d_len,
                              double *extrm_val);

// assume we have d_data as a device pointer with our data, of length data_len
template <int SIGN, int elem_per_thread = 2, int thread_per_block = 256>
__host__ double cuda_find_max(const double *d_data, const int data_len)