// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2012 NICTA (www.nicta.com.au)
// Copyright (C) 2012 Ryan Curtin
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_trace
//! @{


//! Immediate trace (sum of diagonal elements) of a square dense matrix
template<typename T1>
arma_hot
arma_warn_unused
inline
typename enable_if2<is_arma_type<T1>::value, typename T1::elem_type>::result
trace(const T1& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const Proxy<T1> A(X);
  
  arma_debug_check( (A.get_n_rows() != A.get_n_cols()), "trace(): matrix must be square sized" );
  
  const uword N = A.get_n_rows();
  
  eT val1 = eT(0);
  eT val2 = eT(0);
  
  uword i,j;
  for(i=0, j=1; j<N; i+=2, j+=2)
    {
    val1 += A.at(i,i);
    val2 += A.at(j,j);
    }
  
  if(i < N)
    {
    val1 += A.at(i,i);
    }
  
  return val1 + val2;
  }



template<typename T1>
arma_hot
arma_warn_unused
inline
typename T1::elem_type
trace(const Op<T1, op_diagmat>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const diagmat_proxy<T1> A(X.m);
  
  const uword N = A.n_elem;
  
  eT val = eT(0);
  
  for(uword i=0; i<N; ++i)
    {
    val += A[i];
    }
  
  return val;
  }



//! speedup for trace(A*B), where the result of A*B is a square sized matrix
template<typename T1, typename T2>
arma_hot
inline
typename T1::elem_type
trace_mul_unwrap(const T1& XA, const T2& XB)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const Proxy<T1>    PA(XA);
  const unwrap<T2> tmpB(XB);
  
  const Mat<eT>& B = tmpB.M;
  
  arma_debug_assert_mul_size(PA.get_n_rows(), PA.get_n_cols(), B.n_rows, B.n_cols, "matrix multiplication");
  
  arma_debug_check( (PA.get_n_rows() != B.n_cols), "trace(): matrix must be square sized" );
  
  const uword N1 = PA.get_n_rows();   // equivalent to B.n_cols, due to square size requirements
  const uword N2 = PA.get_n_cols();   // equivalent to B.n_rows, due to matrix multiplication requirements
  
  eT val = eT(0);
  
  for(uword i=0; i<N1; ++i)
    {
    const eT* B_colmem = B.colptr(i);
    
    eT acc1 = eT(0);
    eT acc2 = eT(0);
    
    uword j,k;
    for(j=0, k=1; k < N2; j+=2, k+=2)
      {
      const eT tmp_j = B_colmem[j];
      const eT tmp_k = B_colmem[k];
      
      acc1 += PA.at(i,j) * tmp_j;
      acc2 += PA.at(i,k) * tmp_k;
      }
    
    if(j < N2)
      {
      acc1 += PA.at(i,j) * B_colmem[j];
      }
    
    val += (acc1 + acc2);
    }
  
  return val;
  }



//! speedup for trace(A*B), where the result of A*B is a square sized matrix
template<typename T1, typename T2>
arma_hot
inline
typename T1::elem_type
trace_mul_proxy(const T1& XA, const T2& XB)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const Proxy<T1> PA(XA);
  const Proxy<T2> PB(XB);
  
  if(is_Mat<typename Proxy<T2>::stored_type>::value == true)
    {
    return trace_mul_unwrap(PA.Q, PB.Q);
    }
  
  arma_debug_assert_mul_size(PA.get_n_rows(), PA.get_n_cols(), PB.get_n_rows(), PB.get_n_cols(), "matrix multiplication");
  
  arma_debug_check( (PA.get_n_rows() != PB.get_n_cols()), "trace(): matrix must be square sized" );
  
  const uword N1 = PA.get_n_rows();   // equivalent to PB.get_n_cols(), due to square size requirements
  const uword N2 = PA.get_n_cols();   // equivalent to PB.get_n_rows(), due to matrix multiplication requirements
  
  eT val = eT(0);
  
  for(uword i=0; i<N1; ++i)
    {
    eT acc1 = eT(0);
    eT acc2 = eT(0);
    
    uword j,k;
    for(j=0, k=1; k < N2; j+=2, k+=2)
      {
      const eT tmp_j = PB.at(j,i);
      const eT tmp_k = PB.at(k,i);
      
      acc1 += PA.at(i,j) * tmp_j;
      acc2 += PA.at(i,k) * tmp_k;
      }
    
    if(j < N2)
      {
      acc1 += PA.at(i,j) * PB.at(j,i);
      }
    
    val += (acc1 + acc2);
    }
  
  return val;
  }



//! speedup for trace(A*B), where the result of A*B is a square sized matrix
template<typename T1, typename T2>
arma_hot
arma_warn_unused
inline
typename T1::elem_type
trace(const Glue<T1, T2, glue_times>& X)
  {
  arma_extra_debug_sigprint();
  
  return (is_Mat<T2>::value) ? trace_mul_unwrap(X.A, X.B) : trace_mul_proxy(X.A, X.B);
  }



//! trace of sparse object
template<typename T1>
arma_hot
arma_warn_unused
inline
typename enable_if2<is_arma_sparse_type<T1>::value, typename T1::elem_type>::result
trace(const T1& x)
  {
  arma_extra_debug_sigprint();
  
  const SpProxy<T1> p(x);
  
  arma_debug_check( (p.get_n_rows() != p.get_n_cols()), "trace(): matrix must be square sized" );
  
  typedef typename T1::elem_type eT;
  
  eT result = eT(0);
  
  typename SpProxy<T1>::const_iterator_type it     = p.begin();
  typename SpProxy<T1>::const_iterator_type it_end = p.end();
  
  while(it != it_end)
    {
    if(it.row() == it.col())
      {
      result += (*it);
      }
    
    ++it;
    }
  
  return result;
  }



//! @}
