#include "dynet/cuda.h"
#include "dynet/gpu-ops.h"
#include "dynet/gpu-kernels.h"
#include "dynet/functors.h"

namespace dynet {
namespace gpu {

// CUDA kernel. Each thread takes care of one element of c
__global__ void ker_dense_to_sparse_assign(int n, const unsigned int *idx, float *src, float *trg) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n)
    trg[idx[id]] = src[id];
}

void dense_to_sparse_assign(int n, const unsigned int *idx, float *src, float *trg) {
  if(n > 0) {
    auto tb = SizeToBlockThreadPair(n);
    int total_size = tb.first*tb.second;
    for(int curr_pos = 0; curr_pos < n; curr_pos += total_size)
      ker_dense_to_sparse_assign<<<tb.first, tb.second>>>(min(total_size, n-curr_pos), idx+curr_pos, src+curr_pos, trg);
  }
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void ker_sparse_to_dense_assign(int n, const unsigned int *idx, float *src, float *trg) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n)
    trg[id] = src[idx[id]];
}

void sparse_to_dense_assign(int n, const unsigned int *idx, float *src, float *trg) {
  if(n > 0) {
    auto tb = SizeToBlockThreadPair(n);
    int total_size = tb.first*tb.second;
    for(int curr_pos = 0; curr_pos < n; curr_pos += total_size)
      ker_sparse_to_dense_assign<<<tb.first, tb.second>>>(min(total_size, n-curr_pos), idx+curr_pos, src, trg+curr_pos);
  }
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void ker_dense_to_sparse_subtract(int n, const unsigned int *idx, float *src, float *trg) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n)
    atomicAdd(trg + idx[id], -src[id]);
}

void dense_to_sparse_subtract(int n, const unsigned int *idx, float *src, float *trg) {
  if(n > 0) {
    auto tb = SizeToBlockThreadPair(n);
    int total_size = tb.first*tb.second;
    for(int curr_pos = 0; curr_pos < n; curr_pos += total_size)
      ker_dense_to_sparse_subtract<<<tb.first, tb.second>>>(min(total_size, n-curr_pos), idx+curr_pos, src+curr_pos, trg);
  }
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void ker_sparse_to_dense_block_assign_and_multiply(int n, const unsigned *idx, int bsize, float mult, float* src, float *trg) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n*bsize)
    trg[id] = src[idx[id/bsize]*bsize+id%bsize] * mult;
}

void sparse_to_dense_block_assign_and_multiply(int n, const unsigned *idx, int bsize, float mult, float *src, float *trg) {
  if(n > 0) {
    auto tb = SizeToBlockThreadPair(n*bsize);
    int total_size = tb.first*tb.second;
    for(int curr_pos = 0; curr_pos < n; curr_pos += total_size/bsize)
      ker_sparse_to_dense_block_assign_and_multiply<<<tb.first, tb.second>>>(min(total_size/bsize, n-curr_pos), idx+curr_pos, bsize, mult, src, trg+curr_pos*bsize);
  }
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void ker_dense_to_sparse_block_add(int n, const unsigned *idx, int bsize, float* src, float *trg) {
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n*bsize)
    atomicAdd(trg + idx[id/bsize]*bsize+id%bsize, src[id]);
}

void dense_to_sparse_block_add(int n, const unsigned *idx, int bsize, float *src, float *trg) {
  if(n > 0) {
    auto tb = SizeToBlockThreadPair(n*bsize);
    int total_size = tb.first*tb.second;
    for(int curr_pos = 0; curr_pos < n; curr_pos += total_size/bsize)
      ker_dense_to_sparse_block_add<<<tb.first, tb.second>>>(min(total_size/bsize, n-curr_pos), idx+curr_pos, bsize, src+curr_pos*bsize, trg);
  }
}

} // namespace gpu
} // namespace dynet
