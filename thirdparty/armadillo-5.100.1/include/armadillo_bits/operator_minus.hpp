// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2012 NICTA (www.nicta.com.au)
// Copyright (C) 2012 Ryan Curtin
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup operator_minus
//! @{



//! unary -
template<typename T1>
arma_inline
typename
enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_neg> >::result
operator-
(const T1& X)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1,eop_neg>(X);
  }



//! cancellation of two consecutive negations: -(-T1)
template<typename T1>
arma_inline
const typename Proxy<T1>::stored_type&
operator-
(const eOp<T1, eop_neg>& X)
  {
  arma_extra_debug_sigprint();
  
  return X.P.Q;
  }



//! Base - scalar
template<typename T1>
arma_inline
typename
enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_scalar_minus_post> >::result
operator-
  (
  const T1&                    X,
  const typename T1::elem_type k
  )
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_scalar_minus_post>(X, k);
  }



//! scalar - Base
template<typename T1>
arma_inline
typename
enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_scalar_minus_pre> >::result
operator-
  (
  const typename T1::elem_type k,
  const T1&                    X
  )
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_scalar_minus_pre>(X, k);
  }



//! complex scalar - non-complex Base
template<typename T1>
arma_inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no),
  const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_pre>
  >::result
operator-
  (
  const std::complex<typename T1::pod_type>& k,
  const T1&                                  X
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_pre>('j', X, k);
  }



//! non-complex Base - complex scalar
template<typename T1>
arma_inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no),
  const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_post>
  >::result
operator-
  (
  const T1&                                  X,
  const std::complex<typename T1::pod_type>& k
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_post>('j', X, k);
  }



//! subtraction of Base objects with same element type
template<typename T1, typename T2>
arma_inline
typename
enable_if2
  <
  is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value,
  const eGlue<T1, T2, eglue_minus>
  >::result
operator-
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return eGlue<T1, T2, eglue_minus>(X, Y);
  }



//! subtraction of Base objects with different element types
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_arma_type<T2>::value && (is_same_type<typename T1::elem_type, typename T2::elem_type>::no)),
  const mtGlue<typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, glue_mixed_minus>
  >::result
operator-
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
  
  return mtGlue<out_eT, T1, T2, glue_mixed_minus>( X, Y );
  }



//! unary "-" for sparse objects 
template<typename T1>
inline
typename
enable_if2
  <
  is_arma_sparse_type<T1>::value && is_signed<typename T1::elem_type>::value,
  SpOp<T1,spop_scalar_times>
  >::result
operator-
(const T1& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  return SpOp<T1,spop_scalar_times>(X, eT(-1));
  }



//! subtraction of two sparse objects
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value && is_arma_sparse_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  const SpGlue<T1,T2,spglue_minus>
  >::result
operator-
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return SpGlue<T1,T2,spglue_minus>(X,Y);
  }



//! subtraction of one sparse and one dense object
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  Mat<typename T1::elem_type>
  >::result
operator-
  (
  const T1& x,
  const T2& y
  )
  {
  arma_extra_debug_sigprint();
  
  const SpProxy<T1> pa(x);
  
  Mat<typename T1::elem_type> result(-y);
  
  arma_debug_assert_same_size( pa.get_n_rows(), pa.get_n_cols(), result.n_rows, result.n_cols, "subtraction" );
  
  typename SpProxy<T1>::const_iterator_type it     = pa.begin();
  typename SpProxy<T1>::const_iterator_type it_end = pa.end();
  
  while(it != it_end)
    {
    result.at(it.row(), it.col()) += (*it);
    ++it;
    }
  
  return result;
  }



//! subtraction of one dense and one sparse object
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_arma_sparse_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  Mat<typename T1::elem_type>
  >::result
operator-
  (
  const T1& x,
  const T2& y
  )
  {
  arma_extra_debug_sigprint();
  
  Mat<typename T1::elem_type> result(x);
  
  const SpProxy<T2> pb(y.get_ref());
  
  arma_debug_assert_same_size( result.n_rows, result.n_cols, pb.get_n_rows(), pb.get_n_cols(), "subtraction" );
  
  typename SpProxy<T2>::const_iterator_type it     = pb.begin();
  typename SpProxy<T2>::const_iterator_type it_end = pb.end();

  while(it != it_end)
    {
    result.at(it.row(), it.col()) -= (*it);
    ++it;
    }
  
  return result;
  }



//! @}
