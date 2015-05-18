// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// Copyright (C) 2012 Ryan Curtin
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_dot
//! @{


template<typename T1, typename T2>
arma_inline
arma_warn_unused
typename
enable_if2
  <
  is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::yes,
  typename T1::elem_type
  >::result
dot
  (
  const T1& A,
  const T2& B
  )
  {
  arma_extra_debug_sigprint();
  
  return op_dot::apply(A,B);
  }



template<typename T1, typename T2>
inline
arma_warn_unused
typename
enable_if2
  <
  is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::no,
  typename promote_type<typename T1::elem_type, typename T2::elem_type>::result
  >::result
dot
  (
  const T1& A,
  const T2& B
  )
  {
  arma_extra_debug_sigprint();
  
  return op_dot_mixed::apply(A,B);
  }



template<typename T1, typename T2>
inline
arma_warn_unused
typename
enable_if2
  <
  is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value,
  typename T1::elem_type
  >::result
norm_dot
  (
  const T1& A, 
  const T2& B
  )
  {
  arma_extra_debug_sigprint();
  
  return op_norm_dot::apply(A,B);
  }



//
// cdot



template<typename T1, typename T2>
arma_inline
arma_warn_unused
typename
enable_if2
  <
  is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value && is_not_complex<typename T1::elem_type>::value,
  typename T1::elem_type
  >::result
cdot
  (
  const T1& A,
  const T2& B
  )
  {
  arma_extra_debug_sigprint();
  
  return op_dot::apply(A,B);
  }




template<typename T1, typename T2>
arma_inline
arma_warn_unused
typename
enable_if2
  <
  is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value && is_complex<typename T1::elem_type>::value,
  typename T1::elem_type
  >::result
cdot
  (
  const T1& A,
  const T2& B
  )
  {
  arma_extra_debug_sigprint();
  
  return op_cdot::apply(A,B);
  }



// convert dot(htrans(x), y) to cdot(x,y)

template<typename T1, typename T2>
arma_inline
arma_warn_unused
typename
enable_if2
  <
  is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value && is_complex<typename T1::elem_type>::value,
  typename T1::elem_type
  >::result
dot
  (
  const Op<T1, op_htrans>& A,
  const T2&                B
  )
  {
  arma_extra_debug_sigprint();
  
  return cdot(A.m, B);
  }



//
// for sparse matrices
// 



namespace priv
  {
  
  template<typename T1, typename T2>
  arma_hot
  inline
  typename T1::elem_type
  dot_helper(const SpProxy<T1>& pa, const SpProxy<T2>& pb)
    {
    typedef typename T1::elem_type eT;
    
    // Iterate over both objects and see when they are the same
    eT result = eT(0);
    
    typename SpProxy<T1>::const_iterator_type a_it  = pa.begin();
    typename SpProxy<T1>::const_iterator_type a_end = pa.end();
    
    typename SpProxy<T2>::const_iterator_type b_it  = pb.begin();
    typename SpProxy<T2>::const_iterator_type b_end = pb.end();
    
    while((a_it != a_end) && (b_it != b_end))
      {
      if(a_it == b_it)
        {
        result += (*a_it) * (*b_it);
        
        ++a_it;
        ++b_it;
        }
      else if((a_it.col() < b_it.col()) || ((a_it.col() == b_it.col()) && (a_it.row() < b_it.row())))
        {
        // a_it is "behind"
        ++a_it;
        }
      else
        {
        // b_it is "behind"
        ++b_it;
        }
      }
    
    return result;
    }
  
  }



//! dot product of two sparse objects
template<typename T1, typename T2>
arma_warn_unused
arma_hot
inline
typename
enable_if2
  <(is_arma_sparse_type<T1>::value) && (is_arma_sparse_type<T2>::value) && (is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
   typename T1::elem_type
  >::result
dot
  (
  const T1& x,
  const T2& y
  )
  {
  arma_extra_debug_sigprint();
  
  const SpProxy<T1> pa(x);
  const SpProxy<T2> pb(y);
  
  arma_debug_assert_same_size(pa.get_n_rows(), pa.get_n_cols(), pb.get_n_rows(), pb.get_n_cols(), "dot()");
  
  typedef typename T1::elem_type eT;
  
  typedef typename SpProxy<T1>::stored_type pa_Q_type;
  typedef typename SpProxy<T2>::stored_type pb_Q_type;
  
  if(
         ( (SpProxy<T1>::must_use_iterator == false) && (SpProxy<T2>::must_use_iterator == false) )
      && ( (is_SpMat<pa_Q_type>::value     == true ) && (is_SpMat<pb_Q_type>::value     == true ) )   
    )
    {
    const unwrap_spmat<pa_Q_type> tmp_a(pa.Q);
    const unwrap_spmat<pb_Q_type> tmp_b(pb.Q);
    
    const SpMat<eT>& A = tmp_a.M;
    const SpMat<eT>& B = tmp_b.M;
    
    if( &A == &B )
      {
      // We can do it directly!
      return op_dot::direct_dot_arma(A.n_nonzero, A.values, A.values);
      }
    else
      {
      return priv::dot_helper(pa,pb);
      }
    }
  else
    {
    return priv::dot_helper(pa,pb);
    }
  }



//! dot product of one dense and one sparse object
template<typename T1, typename T2>
arma_warn_unused
arma_hot
inline
typename
enable_if2
  <(is_arma_type<T1>::value) && (is_arma_sparse_type<T2>::value) && (is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
   typename T1::elem_type
  >::result
dot
  (
  const T1& x,
  const T2& y
  )
  {
  arma_extra_debug_sigprint();
  
  const   Proxy<T1> pa(x);
  const SpProxy<T2> pb(y);
  
  arma_debug_assert_same_size(pa.get_n_rows(), pa.get_n_cols(), pb.get_n_rows(), pb.get_n_cols(), "dot()");
  
  typedef typename T1::elem_type eT;
  
  eT result = eT(0);
  
  typename SpProxy<T2>::const_iterator_type it     = pb.begin();
  typename SpProxy<T2>::const_iterator_type it_end = pb.end();
  
  // prefer_at_accessor won't save us operations
  while(it != it_end)
    {
    result += (*it) * pa.at(it.row(), it.col());
    ++it;
    }
  
  return result;
  }



//! dot product of one sparse and one dense object
template<typename T1, typename T2>
arma_warn_unused
arma_hot
inline
typename
enable_if2
  <(is_arma_sparse_type<T1>::value) && (is_arma_type<T2>::value) && (is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
   typename T1::elem_type
  >::result
dot
  (
  const T1& x,
  const T2& y
  )
  {
  arma_extra_debug_sigprint();
  
  // this is commutative
  return dot(y, x);
  }



//! @}
