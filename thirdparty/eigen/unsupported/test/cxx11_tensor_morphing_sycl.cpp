// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
// Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_morphing_sycl
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_SYCL


#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

template <typename DataType, int DataLayout>
static void test_simple_slice(const Eigen::SyclDevice &sycl_device)
{
  int sizeDim1 = 2;
  int sizeDim2 = 3;
  int sizeDim3 = 5;
  int sizeDim4 = 7;
  int sizeDim5 = 11;
  array<int, 5> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4, sizeDim5}};
  Tensor<DataType, 5,DataLayout> tensor(tensorRange);
  tensor.setRandom();
  array<int, 5> slice1_range ={{1, 1, 1, 1, 1}};
  Tensor<DataType, 5,DataLayout> slice1(slice1_range);

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(slice1.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 5,DataLayout>> gpu1(gpu_data1, tensorRange);
  TensorMap<Tensor<DataType, 5,DataLayout>> gpu2(gpu_data2, slice1_range);
  Eigen::DSizes<ptrdiff_t, 5> indices(1,2,3,4,5);
  Eigen::DSizes<ptrdiff_t, 5> sizes(1,1,1,1,1);
  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
  gpu2.device(sycl_device)=gpu1.slice(indices, sizes);
  sycl_device.memcpyDeviceToHost(slice1.data(), gpu_data2,(slice1.size())*sizeof(DataType));
  VERIFY_IS_EQUAL(slice1(0,0,0,0,0), tensor(1,2,3,4,5));


  array<int, 5> slice2_range ={{1,1,2,2,3}};
  Tensor<DataType, 5,DataLayout> slice2(slice2_range);
  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(slice2.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 5,DataLayout>> gpu3(gpu_data3, slice2_range);
  Eigen::DSizes<ptrdiff_t, 5> indices2(1,1,3,4,5);
  Eigen::DSizes<ptrdiff_t, 5> sizes2(1,1,2,2,3);
  gpu3.device(sycl_device)=gpu1.slice(indices2, sizes2);
  sycl_device.memcpyDeviceToHost(slice2.data(), gpu_data3,(slice2.size())*sizeof(DataType));
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        VERIFY_IS_EQUAL(slice2(0,0,i,j,k), tensor(1,1,3+i,4+j,5+k));
      }
    }
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
  sycl_device.deallocate(gpu_data3);
}

template<typename DataType, typename dev_Selector> void sycl_slicing_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_simple_slice<DataType, RowMajor>(sycl_device);
  test_simple_slice<DataType, ColMajor>(sycl_device);
}
void test_cxx11_tensor_morphing_sycl()
{
  /// Currentlly it only works on cpu. Adding GPU cause LLVM ERROR in cunstructing OpenCL Kernel at runtime.
//	printf("Test on GPU: OpenCL\n");
//	CALL_SUBTEST(sycl_device_test_per_device((cl::sycl::gpu_selector())));
  printf("repeating the test on CPU: OpenCL\n");
  CALL_SUBTEST(sycl_slicing_test_per_device<float>((cl::sycl::cpu_selector())));
  printf("repeating the test on CPU: HOST\n");
  CALL_SUBTEST(sycl_slicing_test_per_device<float>((cl::sycl::host_selector())));
  printf("Test Passed******************\n" );


}
