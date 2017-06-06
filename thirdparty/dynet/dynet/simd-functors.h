#ifndef DYNET_XFUNCTORS_H
#define DYNET_XFUNCTORS_H

#ifndef __CUDACC__
#include <Eigen/Eigen>
#endif

#include "dynet/functors.h"

// these functors are implemented to exploit Eigen's internal logic for doing
// vectorized arithmetic. I'm putting them in a separate file since, if Eigen
// breaks backward compatibility by changing an internal interface, I want
// the necessary changes to be localized.
//
// to implement your own functor, you need to provide
//   1) operator() implemented on the scalar data type
//   2) packetOp implemented using vector ("packet") type
//   3) the functor_traits specialization for your functor
//      that tells the compiler whether your architecture
//      has vectorized support for the operations you need
//      and an estimate of the cost of the operation

namespace dynet {
template<typename Scalar> struct const_add_op {
  const_add_op(const Scalar& c) : c(c) {}
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
    return c + x;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    using namespace Eigen::internal;
    return padd(pset1<Packet>(c), x);
  }
  Scalar c;
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::const_add_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2,
    PacketAccess = packet_traits<Scalar>::HasAdd
  };
};
} }

namespace dynet {
template<typename Scalar> struct const_minus_op {
  const_minus_op(const Scalar& c) : c(c) {}
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
    return c - x;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    using namespace Eigen::internal;
    return psub(pset1<Packet>(c), x);
  }
  Scalar c;
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::const_minus_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2,
    PacketAccess = packet_traits<Scalar>::HasSub
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_logistic_sigmoid_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_logistic_sigmoid_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
    using std::exp;
    const Scalar one = Scalar(1);
    return one / (one + exp(-x));
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pdiv(one, padd(one, pexp(pnegate(x))));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_logistic_sigmoid_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 6,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasDiv &&
                   packet_traits<Scalar>::HasNegate && packet_traits<Scalar>::HasExp
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_erf_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_erf_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    using std::exp;
    const Scalar sqrt_pi_over2(1.1283791670955125738961589);
    return sqrt_pi_over2 * exp(-x * x) * d;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet sqrt_pi_over2 = pset1<Packet>(1.1283791670955125738961589);
    return pmul(sqrt_pi_over2, pmul(pexp(pnegate(pmul(x, x))), d));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_erf_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::MulCost * 8,
    PacketAccess = packet_traits<Scalar>::HasExp && packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasNegate
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_logistic_sigmoid_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_logistic_sigmoid_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& t, const Scalar& d) const {
    const Scalar one = Scalar(1);
    return (one - t) * t * d;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(psub(one, t), pmul(t, d));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_logistic_sigmoid_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + NumTraits<Scalar>::MulCost * 2,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_tanh_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tanh_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const { using std::tanh; return tanh(a); }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& a) const { return Eigen::internal::ptanh(a); }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_tanh_op<Scalar> > {
  enum {
    Cost = 5 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasTanh
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_tanh_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tanh_backward_op)
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& t, const Scalar& d) const { return (1 - t * t) * d; }
  template<typename Packet>
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(psub(one, pmul(t, t)), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_tanh_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + 2 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul
  };
};
}}

namespace dynet {
//this is slower than the dumb implementation, probably because of the pset operations
// which could be factored out into the constructor, but the Packet type isn't used
// then (and I think fixing this would be hard)
template<typename Scalar> struct scalar_nlsoftmax_backward_op {
  scalar_nlsoftmax_backward_op(const Scalar& lz, const Scalar& err) : logz(lz), d(err) {}
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& t) const {
    using std::exp;
    return exp(t - logz) * d;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& t) const {
    using namespace Eigen::internal;
    const Packet lz = pset1<Packet>(logz);
    const Packet dd = pset1<Packet>(d);
    return pmul(pexp(psub(t, lz)), dd);
  }
  Scalar logz;
  Scalar d;
};}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_nlsoftmax_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + 6 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasExp
  };
};
}}

#endif
