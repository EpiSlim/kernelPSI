// C/C++ headers
#include <cmath>
#include <float.h>

// CUDA headers
#include "cublas_v2.h"
#include "curand.h"
#include <cuda.h>
#include <cuda_runtime.h>

template <int SIGN, int BLKSZ>
__global__ void cuda_max_reduce(const double *d_data, const int d_len,
                                double *extrm_val) {
  volatile __shared__ double smem[BLKSZ];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  // starting index for each block to begin loading the input data into shared
  // memory
  const int bid_sidx = bid * BLKSZ;

// load the input data to smem, with padding if needed. each thread handles 2
// elements
#pragma unroll
  for (int i = 0; i < 2; i++) {
    // get the index for the thread to load into shared memory
    const int tid_idx = 2 * tid + i;
    const int ld_idx = bid_sidx + tid_idx;
    if (ld_idx < (bid + 1) * BLKSZ && ld_idx < d_len)
      smem[tid_idx] = d_data[ld_idx];
    else
      smem[tid_idx] = -SIGN * DBL_MAX;

    __syncthreads();
  }

  // run the reduction per-block
  for (unsigned int stride = BLKSZ / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] =
          ((SIGN * smem[tid] > SIGN * smem[tid + stride]) ? smem[tid]
                                                          : smem[tid + stride]);
    }
    __syncthreads();
  }

  // write the per-block result out from shared memory to global memory
  extrm_val[bid] = smem[0];
}

// assume we have d_data as a device pointer with our data, of length data_len
template <int SIGN, int elem_per_thread = 2, int thread_per_block = 256>
__host__ double cuda_find_max(const double *d_data, const int data_len) {
  // in your host code, invoke the kernel with something along the lines of:
  const int BLKSZ = elem_per_thread *
                    thread_per_block; // number of elements to process per block
  const int blocks_per_grid = ceil((float)data_len / (BLKSZ));

  double *d_extrm;
  cudaMallocManaged((void **)&d_extrm, sizeof(double) * blocks_per_grid);

  cuda_max_reduce<BLKSZ, SIGN>
      <<<blocks_per_grid, thread_per_block>>>(d_data, data_len, d_extrm);

  cudaDeviceSynchronize();

  double result = d_extrm[0];
  for (int i = 1; i < blocks_per_grid; i++) 
     result = ((SIGN * d_extrm[i] > SIGN * result) ? d_extrm[i] : result);

  cudaFree(d_extrm);

  return result;
}
