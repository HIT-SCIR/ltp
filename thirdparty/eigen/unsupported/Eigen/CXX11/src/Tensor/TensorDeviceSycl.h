// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>

//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_SYCL) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H)
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H

namespace Eigen {

#define ConvertToActualTypeSycl(T, buf_acc) reinterpret_cast<typename cl::sycl::global_ptr<T>::pointer_t>((&(*buf_acc.get_pointer())))

struct QueueInterface {
  /// class members:
  bool exception_caught_ = false;

  mutable std::mutex mutex_;

  /// std::map is the container used to make sure that we create only one buffer
  /// per pointer. The lifespan of the buffer now depends on the lifespan of SyclDevice.
  /// If a non-read-only pointer is needed to be accessed on the host we should manually deallocate it.
  mutable std::map<const uint8_t *, cl::sycl::buffer<uint8_t, 1>> buffer_map;
  /// sycl queue
  mutable cl::sycl::queue m_queue;
  /// creating device by using selector
  /// SyclStreamDevice is not owned. it is the caller's responsibility to destroy it.
  template<typename dev_Selector> explicit QueueInterface(dev_Selector s):
#ifdef EIGEN_EXCEPTIONS
  m_queue(cl::sycl::queue(s, [&](cl::sycl::exception_list l) {
    for (const auto& e : l) {
      try {
        if (e) {
           exception_caught_ = true;
           std::rethrow_exception(e);
        }
      } catch (cl::sycl::exception e) {
        std::cerr << e.what() << std::endl;
      }
    }
  }))
#else
  m_queue(cl::sycl::queue(s))
#endif
  {}

  /// creating device by using selector
  /// SyclStreamDevice is not owned. it is the caller's responsibility to destroy it.
  explicit QueueInterface(cl::sycl::device d):
#ifdef EIGEN_EXCEPTIONS
  m_queue(cl::sycl::queue(d, [&](cl::sycl::exception_list l) {
	for (const auto& e : l) {
	  try {
	    if (e) {
	      exception_caught_ = true;
	      std::rethrow_exception(e);
	    }
	  } catch (cl::sycl::exception e) {
	    std::cerr << e.what() << std::endl;
	  }
	}
      }))
#else
  m_queue(cl::sycl::queue(d))
#endif
  {}


  /// Allocating device pointer. This pointer is actually an 8 bytes host pointer used as key to access the sycl device buffer.
  /// The reason is that we cannot use device buffer as a pointer as a m_data in Eigen leafNode expressions. So we create a key
  /// pointer to be used in Eigen expression construction. When we convert the Eigen construction into the sycl construction we
  /// use this pointer as a key in our buffer_map and we make sure that we dedicate only one buffer only for this pointer.
  /// The device pointer would be deleted by calling deallocate function.
  EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    auto buf = cl::sycl::buffer<uint8_t,1>(cl::sycl::range<1>(num_bytes));
    auto ptr =buf.get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::host_buffer>().get_pointer();
    buf.set_final_data(nullptr);
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_map.insert(std::pair<const uint8_t *, cl::sycl::buffer<uint8_t, 1>>(ptr,buf));
    return static_cast<void*>(ptr);
  }

  /// This is used to deallocate the device pointer. p is used as a key inside
  /// the map to find the device buffer and delete it.
  EIGEN_STRONG_INLINE void deallocate(const void *p) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffer_map.find(static_cast<const uint8_t*>(p));
    if (it != buffer_map.end()) {
      buffer_map.erase(it);
    }
  }

  EIGEN_STRONG_INLINE void deallocate_all() const {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_map.clear();
  }

  EIGEN_STRONG_INLINE std::map<const uint8_t *, cl::sycl::buffer<uint8_t,1>>::iterator find_buffer(const void* ptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it1 = buffer_map.find(static_cast<const uint8_t*>(ptr));
    if (it1 != buffer_map.end()){
      return it1;
    }
    else{
      for(std::map<const uint8_t *, cl::sycl::buffer<uint8_t,1>>::iterator it=buffer_map.begin(); it!=buffer_map.end(); ++it){
        auto size = it->second.get_size();
        if((it->first <  (static_cast<const uint8_t*>(ptr))) && ((static_cast<const uint8_t*>(ptr)) < (it->first + size)) ) return it;
      }
    }
    std::cerr << "No sycl buffer found. Make sure that you have allocated memory for your buffer by calling allocate function in SyclDevice"<< std::endl;
    abort();
  }

  // This function checks if the runtime recorded an error for the
  // underlying stream device.
  EIGEN_STRONG_INLINE bool ok() const {
    if (!exception_caught_) {
      m_queue.throw_asynchronous();
    }
    return !exception_caught_;
  }
  // destructor
  ~QueueInterface() { buffer_map.clear(); }
};

template <typename T> class MemCopyFunctor {
 public:
  typedef cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> read_accessor;
  typedef cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer> write_accessor;
  MemCopyFunctor(read_accessor src_acc, write_accessor dst_acc, size_t rng, size_t i, size_t offset): m_src_acc(src_acc), m_dst_acc(dst_acc), m_rng(rng), m_i(i), m_offset(offset) {}
  void operator()(cl::sycl::nd_item<1> itemID) {
    auto src_ptr = ConvertToActualTypeSycl(T, m_src_acc);
    auto dst_ptr = ConvertToActualTypeSycl(T, m_dst_acc);
    auto globalid = itemID.get_global_linear_id();
    if (globalid < m_rng) {
      dst_ptr[globalid + m_i] = src_ptr[globalid + m_offset];
    }
  }
 private:
  read_accessor m_src_acc;
  write_accessor m_dst_acc;
  size_t m_rng;
  size_t m_i;
  size_t m_offset;
};

struct SyclDevice {
  // class member.
  QueueInterface* m_queue_stream;
  /// QueueInterface is not owned. it is the caller's responsibility to destroy it.
  explicit SyclDevice(QueueInterface* queue_stream) : m_queue_stream(queue_stream){}

  /// Creation of sycl accessor for a buffer. This function first tries to find
  /// the buffer in the buffer_map. If found it gets the accessor from it, if not,
  /// the function then adds an entry by creating a sycl buffer for that particular pointer.
  template <cl::sycl::access::mode AcMd> EIGEN_STRONG_INLINE cl::sycl::accessor<uint8_t, 1, AcMd, cl::sycl::access::target::global_buffer>
  get_sycl_accessor(size_t num_bytes, cl::sycl::handler &cgh, const void* ptr) const {
    return (get_sycl_buffer(num_bytes, ptr).template get_access<AcMd, cl::sycl::access::target::global_buffer>(cgh));
  }

  /// Accessing the created sycl device buffer for the device pointer
  EIGEN_STRONG_INLINE cl::sycl::buffer<uint8_t, 1>& get_sycl_buffer(size_t , const void * ptr) const {
    return m_queue_stream->find_buffer(ptr)->second;
  }

  /// This is used to prepare the number of threads and also the number of threads per block for sycl kernels
  EIGEN_STRONG_INLINE void parallel_for_setup(size_t n, size_t &tileSize, size_t &rng, size_t &GRange)  const {
      tileSize =sycl_queue().get_device(). template get_info<cl::sycl::info::device::max_work_group_size>()/2;
      rng = n;
      if (rng==0) rng=1;
      GRange=rng;
      if (tileSize>GRange) tileSize=GRange;
      else if(GRange>tileSize){
        size_t xMode = GRange % tileSize;
        if (xMode != 0) GRange += (tileSize - xMode);
      }
    }
  /// allocate device memory
  EIGEN_STRONG_INLINE void *allocate(size_t num_bytes) const {
      return m_queue_stream->allocate(num_bytes);
  }
  /// deallocate device memory
  EIGEN_STRONG_INLINE void deallocate(const void *p) const {
     m_queue_stream->deallocate(p);
   }

  // some runtime conditions that can be applied here
  EIGEN_STRONG_INLINE bool isDeviceSuitable() const { return true; }


  /// the memcpy function
  template<typename T> EIGEN_STRONG_INLINE void memcpy(void *dst, const T *src, size_t n) const {
    auto it1 = m_queue_stream->find_buffer((void*)src);
    auto it2 = m_queue_stream->find_buffer(dst);
    auto offset= (static_cast<const uint8_t*>(static_cast<const void*>(src))) - it1->first;
    auto i= (static_cast<const uint8_t*>(dst)) - it2->first;
    offset/=sizeof(T);
    i/=sizeof(T);
    size_t rng, GRange, tileSize;
    parallel_for_setup(n/sizeof(T), tileSize, rng, GRange);
    sycl_queue().submit([&](cl::sycl::handler &cgh) {
      auto src_acc =it1->second.template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>(cgh);
      auto dst_acc =it2->second.template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>(cgh);
      cgh.parallel_for(cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange), cl::sycl::range<1>(tileSize)), MemCopyFunctor<T>(src_acc, dst_acc, rng, 0, offset));
    });
    sycl_queue().throw_asynchronous();
  }

  /// The memcpyHostToDevice is used to copy the device only pointer to a host pointer. Using the device
  /// pointer created as a key we find the sycl buffer and get the host accessor with discard_write mode
  /// on it. Using a discard_write accessor guarantees that we do not bring back the current value of the
  /// buffer to host. Then we use the memcpy to copy the data to the host accessor. The first time that
  /// this buffer is accessed, the data will be copied to the device.
  template<typename T> EIGEN_STRONG_INLINE void memcpyHostToDevice(T *dst, const T *src, size_t n) const {
    auto host_acc= get_sycl_buffer(n, dst). template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::host_buffer>();
    ::memcpy(host_acc.get_pointer(), src, n);
  }
  /// The memcpyDeviceToHost is used to copy the data from host to device. Here, in order to avoid double copying the data. We create a sycl
  /// buffer with map_allocator for the destination pointer with a discard_write accessor on it. The lifespan of the buffer is bound to the
  /// lifespan of the memcpyDeviceToHost function. We create a kernel to copy the data, from the device- only source buffer to the destination
  /// buffer with map_allocator on the gpu in parallel. At the end of the function call the destination buffer would be destroyed and the data
  /// would be available on the dst pointer using fast copy technique (map_allocator). In this case we can make sure that we copy the data back
  /// to the cpu only once per function call.
  template<typename T> EIGEN_STRONG_INLINE void memcpyDeviceToHost(void *dst, const T *src, size_t n) const {
    auto it = m_queue_stream->find_buffer(src);
    auto offset =static_cast<const uint8_t*>(static_cast<const void*>(src))- it->first;
    offset/=sizeof(T);
    size_t rng, GRange, tileSize;
    parallel_for_setup(n/sizeof(T), tileSize, rng, GRange);
    // Assuming that the dst is the start of the destination pointer
    auto dest_buf = cl::sycl::buffer<uint8_t, 1, cl::sycl::map_allocator<uint8_t> >(static_cast<uint8_t*>(dst), cl::sycl::range<1>(rng*sizeof(T)));
    sycl_queue().submit([&](cl::sycl::handler &cgh) {
      auto src_acc= it->second.template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>(cgh);
      auto dst_acc =dest_buf.template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>(cgh);
      cgh.parallel_for( cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange), cl::sycl::range<1>(tileSize)), MemCopyFunctor<T>(src_acc, dst_acc, rng, 0, offset));
    });
    sycl_queue().throw_asynchronous();
  }
  /// returning the sycl queue
  EIGEN_STRONG_INLINE cl::sycl::queue& sycl_queue() const { return m_queue_stream->m_queue;}
  /// Here is the implementation of memset function on sycl.
  template<typename T>  EIGEN_STRONG_INLINE void memset(T *buff, int c, size_t n) const {
    size_t rng, GRange, tileSize;
    parallel_for_setup(n/sizeof(T), tileSize, rng, GRange);
    sycl_queue().submit([&](cl::sycl::handler &cgh) {
      auto buf_acc =get_sycl_buffer(n, static_cast<uint8_t*>(static_cast<void*>(buff))). template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>(cgh);
      cgh.parallel_for<SyclDevice>( cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange), cl::sycl::range<1>(tileSize)), [=](cl::sycl::nd_item<1> itemID) {
        auto globalid=itemID.get_global_linear_id();
        if (globalid< buf_acc.get_size()) {
          for(size_t i=0; i<sizeof(T); i++)
            buf_acc[globalid*sizeof(T) + i] = c;
        }
      });
    });
    sycl_queue().throw_asynchronous();
  }
  /// No need for sycl it should act the same as CPU version
  EIGEN_STRONG_INLINE int majorDeviceVersion() const { return 1; }
  /// There is no need to synchronise the buffer in sycl as it is automatically handled by sycl runtime scheduler.
  EIGEN_STRONG_INLINE void synchronize() const {
    sycl_queue().wait_and_throw();
  }
  // This function checks if the runtime recorded an error for the
  // underlying stream device.
  EIGEN_STRONG_INLINE bool ok() const {
    return m_queue_stream->ok();
  }
};


}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H
