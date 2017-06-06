#ifndef DYNET_CUDNN_TYPES_H_
#define DYNET_CUDNN_TYPES_H_

#include "dynet/dynet.h"
#include "dynet/cuda.h"

#if HAVE_CUDNN
template <class T>
struct DataTypeToCudnnType {};

#define MATCH_TYPE_TO_CUDNN_TYPE(TYPE, ENUM)   \
  template <>                                  \
  struct DataTypeToCudnnType<TYPE> {           \
    static const cudnnDataType_t value = ENUM; \
  }

MATCH_TYPE_TO_CUDNN_TYPE(float, CUDNN_DATA_FLOAT);
MATCH_TYPE_TO_CUDNN_TYPE(double, CUDNN_DATA_DOUBLE);

#undef MATCH_TYPE_TO_CUDNN_TYPE
#endif

namespace dynet {

// A helper class to allocate memory from the aux_mem pointer for complex operators
// e.g. Conv2D
struct NodeMemPool {
 public:
  explicit NodeMemPool() : capacity_(0), used_(0), mem_(NULL) {}
  explicit NodeMemPool(const int capacity, void* mem) 
      : capacity_(capacity), used_(0), mem_(mem) {}

  void* allocate(size_t nbytes) {
    if (used_ + nbytes > capacity_) {
      std::ostringstream oss; oss  
          << "aux_mem_pool allocate memory failed: exceed maximally allowed size";
      throw std::runtime_error(oss.str());
    }
    void* res = static_cast<char*>(mem_) + used_;
    used_ += nbytes;
    return res;
  }

  void free() {
    used_ = 0;
  }

  void* head() {
    return mem_;
  }

  size_t size() {
    return capacity_;
  }

 private:
  size_t capacity_;
  size_t used_;
  void* mem_;
};

} // namespace dynet

#endif
