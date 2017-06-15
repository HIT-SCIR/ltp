// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Cummins Chris PhD student at The University of Edinburgh.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*****************************************************************
 * TensorSyclRun.h
 *
 * \brief:
 * Schedule_kernel invoke an specialised version of kernel struct. The
 * specialisation is based on the data dimension in sycl buffer
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_SYCLRUN_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_SYCLRUN_HPP

namespace Eigen {
namespace TensorSycl {
/// The run function in tensor sycl convert the expression tree to a buffer
/// based expression tree;
/// creates the expression tree for the device with accessor to buffers;
/// construct the kernel and submit it to the sycl queue.
template <typename Expr, typename Dev>
void run(Expr &expr, Dev &dev) {
  Eigen::TensorEvaluator<Expr, Dev> evaluator(expr, dev);
  const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
  if (needs_assign) {
    typedef  typename internal::createPlaceHolderExpression<Expr>::Type PlaceHolderExpr;
    auto functors = internal::extractFunctors(evaluator);

    dev.sycl_queue().submit([&](cl::sycl::handler &cgh) {
      // create a tuple of accessors from Evaluator
      auto tuple_of_accessors = internal::createTupleOfAccessors<decltype(evaluator)>(cgh, evaluator);
      size_t range, GRange, tileSize;
      dev.parallel_for_setup(utility::tuple::get<0>(tuple_of_accessors).get_range()[0]/sizeof(typename Expr::Scalar), tileSize, range, GRange);

      // run the kernel
      cgh.parallel_for<PlaceHolderExpr>( cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange), cl::sycl::range<1>(tileSize)), [=](cl::sycl::nd_item<1> itemID) {
        typedef  typename internal::ConvertToDeviceExpression<Expr>::Type DevExpr;
        auto device_expr =internal::createDeviceExpression<DevExpr, PlaceHolderExpr>(functors, tuple_of_accessors);
        auto device_evaluator = Eigen::TensorEvaluator<decltype(device_expr.expr), Eigen::DefaultDevice>(device_expr.expr, Eigen::DefaultDevice());
        if (itemID.get_global_linear_id() < range) {
          device_evaluator.evalScalar(static_cast<typename DevExpr::Index>(itemID.get_global_linear_id()));
        }
      });
    });
      dev.sycl_queue().throw_asynchronous();
  }

  evaluator.cleanup();
}
}  // namespace TensorSycl
}  // namespace Eigen

#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_SYCLRUN_HPP
