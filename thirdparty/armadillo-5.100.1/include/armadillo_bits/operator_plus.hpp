// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2012 NICTA (www.nicta.com.au)
// Copyright (C) 2012 Ryan Curtin
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup operator_plus
//! @{



//! unary plus operation (does nothing, but is required for completeness)
template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const T1& >::result
operator+
(const T1& X)
  {
  arma_extra_debug_sigprint();
  
  return X;
  }



//! Base + scalar
template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_scalar_plus> >::result
operator+
(const T1& X, const typename T1::elem_type k)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_scalar_plus>(X, k);
  }



//! scalar + Base
template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_scalar_plus> >::result
operator+
(const typename T1::elem_type k, const T1& X)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_scalar_plus>(X, k);  // NOTE: order is swapped
  }



//! non-complex Base + complex scalar
template<typename T1>
arma_inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no),
  const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_plus>
  >::result
operator+
  (
  const T1&                                  X,
  const std::complex<typename T1::pod_type>& k
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_plus>('j', X, k);
  }



//! complex scalar + non-complex Base
template<typename T1>
arma_inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no),
  const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_plus>
  >::result
operator+
  (
  const std::complex<typename T1::pod_type>& k,
  const T1&                                  X
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_plus>('j', X, k);  // NOTE: order is swapped
  }



//! addition of user-accessible Armadillo objects with same element type
template<typename T1, typename T2>
arma_inline
typename
enable_if2
  <
  is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value,
  const eGlue<T1, T2, eglue_plus>
  >::result
operator+
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return eGlue<T1, T2, eglue_plus>(X, Y);
  }



//! addition of user-accessible Armadillo objects with different element types
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_arma_type<T2>::value && (is_same_type<typename T1::elem_type, typename T2::elem_type>::no)),
  const mtGlue<typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, glue_mixed_plus>
  >::result
operator+
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  return mtGlue<out_eT, T1, T2, glue_mixed_plus>( X, Y );
  }



//! addition of two sparse objects
template<typename T1, typename T2>
inline
arma_hot
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value && is_arma_sparse_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  SpGlue<T1,T2,spglue_plus>
  >::result
operator+
  (
  const T1& x,
  const T2& y
  )
  {
  arma_extra_debug_sigprint();
  
  return SpGlue<T1,T2,spglue_plus>(x, y);
  }



//! addition of sparse and non-sparse object
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_arma_sparse_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  Mat<typename T1::elem_type>
  >::result
operator+
  (
  const T1& x,
  const T2& y
  )
  {
  arma_extra_debug_sigprint();
  
  Mat<typename T1::elem_type> result(x);
  
  const SpProxy<T2> pb(y);
  
  arma_debug_assert_same_size( result.n_rows, result.n_cols, pb.get_n_rows(), pb.get_n_cols(), "addition" );
  
  typename SpProxy<T2>::const_iterator_type it     = pb.begin();
  typename SpProxy<T2>::const_iterator_type it_end = pb.end();
  
  while(it != it_end)
    {
    result.at(it.row(), it.col()) += (*it);
    ++it;
    }
  
  return result;
  }



//! addition of sparse and non-sparse object
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  Mat<typename T1::elem_type>
  >::result
operator+
  (
  const T1& x,
  const T2& y
  )
  {
  arma_extra_debug_sigprint();
  
  // Just call the other order (these operations are commutative)
  return (y + x);
  }



//! @}
