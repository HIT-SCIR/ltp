#include "dynet/mem.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#if !_WINDOWS
#include <sys/shm.h>
#include <sys/mman.h>
#endif

#include <fcntl.h>
#if !_WINDOWS
#include <mm_malloc.h>
#endif
#include "dynet/except.h"
#if HAVE_CUDA
#include "dynet/cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace std;

namespace dynet {

MemAllocator::~MemAllocator() {}

void* CPUAllocator::malloc(size_t n) {
  void* ptr = _mm_malloc(n, align);
  if (!ptr) {
    cerr << "CPU memory allocation failed n=" << n << " align=" << align << endl;
    throw dynet::out_of_memory("CPU memory allocation failed");
  }
  return ptr;
}

void CPUAllocator::free(void* mem) {
  _mm_free(mem);
}

void CPUAllocator::zero(void* p, size_t n) {
  memset(p, 0, n);
}

void* SharedAllocator::malloc(size_t n) {
#if _WINDOWS
  cerr << "Shared memory not supported in Windows" << endl;
  throw dynet::out_of_memory("Shared memory allocation failed");
#else
  void* ptr = mmap(NULL, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_SHARED, -1, 0);
  if (!ptr) {
    cerr << "Shared memory allocation failed n=" << n << endl;
    throw dynet::out_of_memory("Shared memory allocation failed");
  }
  return ptr;
#endif
}

void SharedAllocator::free(void* mem) {
//  munmap(mem, n);
}

void SharedAllocator::zero(void* p, size_t n) {
  memset(p, 0, n);
}

#if HAVE_CUDA
void* GPUAllocator::malloc(size_t n) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaSetDevice(devid));
  CUDA_CHECK(cudaMalloc(&ptr, n));
  if (!ptr) {
    cerr << "GPU memory allocation failed n=" << n << endl;
    throw dynet::out_of_memory("GPU memory allocation failed");
  }
  return ptr;
}

void GPUAllocator::free(void* mem) {
  CUDA_CHECK(cudaFree(mem));
}

void GPUAllocator::zero(void* p, size_t n) {
  CUDA_CHECK(cudaSetDevice(devid));
  CUDA_CHECK(cudaMemsetAsync(p, 0, n));
}

#endif

} // namespace dynet
