#ifndef DYNET_MEM_H
#define DYNET_MEM_H

#include <vector>

namespace dynet {

// allocates memory from the device (CPU, GPU)
// only used to create the memory pools
// creates alignment appropriate for that device
struct MemAllocator {
  explicit MemAllocator(int align) : align(align) {}
  MemAllocator(const MemAllocator&) = delete;
  MemAllocator& operator=(const MemAllocator&) = delete;
  virtual ~MemAllocator();
  virtual void* malloc(std::size_t n) = 0;
  virtual void free(void* mem) = 0;
  virtual void zero(void* p, std::size_t n) = 0;
  inline std::size_t round_up_align(std::size_t n) const {
    if (align < 2) return n;
    return ((n + align - 1) / align) * align;
  }
  const int align;
};

struct CPUAllocator : public MemAllocator {
  CPUAllocator() : MemAllocator(32) {}
  void* malloc(std::size_t n) override;
  void free(void* mem) override;
  void zero(void* p, std::size_t n) override;
};

struct SharedAllocator : public MemAllocator {
  SharedAllocator() : MemAllocator(32) {}
  void* malloc(std::size_t n) override;
  void free(void* mem) override;
  void zero(void* p, std::size_t n) override;
};

#if HAVE_CUDA
struct GPUAllocator : public MemAllocator {
  explicit GPUAllocator(int devid) : MemAllocator(256), devid(devid) {}
  void* malloc(std::size_t n) override;
  void free(void* mem) override;
  void zero(void* p, std::size_t n) override;
  const int devid;
};
#endif

} // namespace dynet

#endif
