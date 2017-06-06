#include "aligned-mem-pool.h"

#include <sstream>

using namespace dynet;

void* InternalMemoryPool::allocate(size_t n) {
  auto rounded_n = a->round_up_align(n);
  if (rounded_n + used > capacity) {
    return 0;
  }
  void* res = static_cast<char*>(mem) + used;
  used += rounded_n;
  return res;
}

void InternalMemoryPool::sys_alloc(size_t cap) {
  capacity = a->round_up_align(cap);
  mem = a->malloc(capacity);
  if (mem == NULL)
    DYNET_RUNTIME_ERR(name << " failed to allocate " << capacity);
  used = 0;
}

AlignedMemoryPool::AlignedMemoryPool(const std::string &name, size_t cap, MemAllocator *a) : name(name), current(0), cap(cap), a(a) {
  DYNET_ASSERT(cap > 0, "Attempt to allocate memory of size 0 in AlignedMemoryPool");
  pools.push_back(new InternalMemoryPool(name, cap, a));
}
AlignedMemoryPool::~AlignedMemoryPool() {
  for ( auto p : pools) { delete p; }
}

void* AlignedMemoryPool::allocate(size_t n) {
  void *res = pools[current]->allocate(n);
  if (res == 0) {
    // round up to the nearest multiple of cap
    pools.push_back(new InternalMemoryPool(name, ((n+cap-1)/cap)*cap, a));
    current++;
    res = pools[current]->allocate(n);
  }
  return res;
}

void AlignedMemoryPool::free() {
  if (current > 0) {
    for (auto p : pools) { delete p; }
    pools.clear();
    pools.push_back(new InternalMemoryPool(name, cap * (current+1), a));
    cap = cap * (current + 1);
    current = 0;
  }
  pools[0]->free();
}

void AlignedMemoryPool::zero_allocated_memory() {
  for (auto p : pools) { p->zero_allocated_memory(); }
}

size_t AlignedMemoryPool::used() {
  if (current == 0) {
    return pools[0]->used;
  }
  size_t res = 0;
  for (auto p : pools) { res += p->used; }
  return res;
}

void AlignedMemoryPool::set_used(size_t s) {
  DYNET_ARG_CHECK(pools.size() == 1, "Dynet does not support both dynamic increasing of memory pool size, and checkpointing functionality in AlignedMemoryPool. If you want to use checkpointing, please pre-allocate enough memory using the --dynet-mem command line option.");
  pools[0]->used = s;
  // TODO: This is disabled for now, because it would require freeing all the memory pools to do properly
  // int c = 0;
  // while (s > pools[c]->used) {
  //   s -= pools[c]->used;
  //   c++;
  //   DYNET_ASSERT(c <= current, "attempt to set_used to a larger value than used().");
  // }
  // // s <= pools[c]->used
  // pools[c]->used = s;
  // current = c;
}
