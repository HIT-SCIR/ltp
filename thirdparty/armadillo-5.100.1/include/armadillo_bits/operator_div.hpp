// Copyright (C) 2009-2012 Conrad Sanderson
// Copyright (C) 2009-2012 NICTA (www.nicta.com.au)
// Copyright (C) 2012 Ryan Curtin
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup operator_div
//! @{



//! Base / scalar
template<typename T1>
arma_inline
typename
enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_scalar_div_post> >::result
operator/
  (
  const T1&                    X,
  const typename T1::elem_type k
  )
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_scalar_div_post>(X, k);
  }



//! scalar / Base
template<typename T1>
arma_inline
typename
enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_scalar_div_pre> >::result
operator/
  (
  const typename T1::elem_type k,
  const T1&                    X
  )
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_scalar_div_pre>(X, k);
  }



//! complex scalar / non-complex Base
template<typename T1>
arma_inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no),
  const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_div_pre>
  >::result
operator/
  (
  const std::complex<typename T1::pod_type>& k,
  const T1&                                  X
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_div_pre>('j', X, k);
  }



//! non-complex Base / complex scalar
template<typename T1>
arma_inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no),
  const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_div_post>
  >::result
operator/
  (
  const T1&                                  X,
  const std::complex<typename T1::pod_type>& k
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_div_post>('j', X, k);
  }



//! element-wise division of Base objects with same element type
template<typename T1, typename T2>
arma_inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  const eGlue<T1, T2, eglue_div>
  >::result
operator/
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return eGlue<T1, T2, eglue_div>(X, Y);
  }



//! element-wise division of Base objects with different element types
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_arma_type<T2>::value && (is_same_type<typename T1::elem_type, typename T2::elem_type>::no)),
  const mtGlue<typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, glue_mixed_div>
  >::result
operator/
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
  
  return mtGlue<out_eT, T1, T2, glue_mixed_div>( X, Y );
  }



//! element-wise division of sparse matrix by scalar
template<typename T1>
inline
typename
enable_if2<is_arma_sparse_type<T1>::value, SpMat<typename T1::elem_type> >::result
operator/
  (
  const T1&                    X,
  const typename T1::elem_type y
  )
  {
  arma_extra_debug_sigprint();
  
  SpMat<typename T1::elem_type> result(X);
  
  result /= y;
  
  return result;
  }



//! element-wise division of one sparse and one dense object
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_sparse_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  SpMat<typename T1::elem_type>
  >::result
operator/
  (
  const T1& x,
  const T2& y
  )
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const SpProxy<T1> pa(x);
  const   Proxy<T2> pb(y);
  
  const uword n_rows = pa.get_n_rows();
  const uword n_cols = pa.get_n_cols();
  
  arma_debug_assert_same_size(n_rows, n_cols, pb.get_n_rows(), pb.get_n_cols(), "element-wise division");
  
  SpMat<eT> result(n_rows, n_cols);
  
  uword new_n_nonzero = 0;
  
  for(uword col=0; col < n_cols; ++col)
  for(uword row=0; row < n_rows; ++row)
    {
    const eT val = pa.at(row,col) / pb.at(row, col);
    
    if(val != eT(0))
      {
      ++new_n_nonzero;
      }
    }
  
  result.mem_resize(new_n_nonzero);
  
  uword cur_pos = 0;
  
  for(uword col=0; col < n_cols; ++col)
  for(uword row=0; row < n_rows; ++row)
    {
    const eT val = pa.at(row,col) / pb.at(row, col);
    
    if(val != eT(0))
      {
      access::rw(result.values[cur_pos]) = val;
      access::rw(result.row_indices[cur_pos]) = row;
      ++access::rw(result.col_ptrs[col + 1]);
      ++cur_pos;
      }
    }
  
  // Fix column pointers
  for(uword col = 1; col <= result.n_cols; ++col)
    {
    access::rw(result.col_ptrs[col]) += result.col_ptrs[col - 1];
    }
  
  return result;
  }



//! element-wise division of one dense and one sparse object
template<typename T1, typename T2>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_arma_sparse_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  Mat<typename T1::elem_type>
  >::result
operator/
  (
  const T1& x,
  const T2& y
  )
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const   Proxy<T1> pa(x);
  const SpProxy<T2> pb(y);
  
  const uword n_rows = pa.get_n_rows();
  const uword n_cols = pa.get_n_cols();
  
  arma_debug_assert_same_size(n_rows, n_cols, pb.get_n_rows(), pb.get_n_cols(), "element-wise division");
  
  Mat<eT> result(n_rows, n_cols);
  
  for(uword col=0; col < n_cols; ++col)
  for(uword row=0; row < n_rows; ++row)
    {
    result.at(row, col) = pa.at(row, col) / pb.at(row, col);
    }
  
  return result;
  }



//! @}
