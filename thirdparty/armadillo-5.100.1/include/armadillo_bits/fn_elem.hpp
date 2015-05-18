// Copyright (C) 2008-2015 Conrad Sanderson
// Copyright (C) 2008-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_elem
//! @{


//
// real

template<typename T1>
arma_inline
typename enable_if2< (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no), const T1& >::result
real(const T1& X)
  {
  arma_extra_debug_sigprint();
  
  return X;
  }



template<typename T1>
arma_inline
const T1&
real(const BaseCube<typename T1::pod_type, T1>& X)
  {
  arma_extra_debug_sigprint();
  
  return X.get_ref();
  }



template<typename T1>
inline
typename enable_if2< (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::yes), const mtOp<typename T1::pod_type, T1, op_real> >::result
real(const T1& X)
  {
  arma_extra_debug_sigprint();
  
  return mtOp<typename T1::pod_type, T1, op_real>( X );
  }



template<typename T1>
inline
const mtOpCube<typename T1::pod_type, T1, op_real>
real(const BaseCube<std::complex<typename T1::pod_type>, T1>& X)
  {
  arma_extra_debug_sigprint();
  
  return mtOpCube<typename T1::pod_type, T1, op_real>( X.get_ref() );
  }



//
// imag

template<typename T1>
inline
const Gen< Mat<typename T1::pod_type>, gen_zeros >
imag(const Base<typename T1::pod_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  const Proxy<T1> A(X.get_ref());
  
  return Gen< Mat<typename T1::pod_type>, gen_zeros>(A.get_n_rows(), A.get_n_cols());
  }



template<typename T1>
inline
const GenCube<typename T1::pod_type, gen_zeros>
imag(const BaseCube<typename T1::pod_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  const ProxyCube<T1> A(X.get_ref());
  
  return GenCube<typename T1::pod_type, gen_zeros>(A.get_n_rows(), A.get_n_cols(), A.get_n_slices());
  }



template<typename T1>
inline
typename enable_if2< (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::yes), const mtOp<typename T1::pod_type, T1, op_imag> >::result
imag(const T1& X)
  {
  arma_extra_debug_sigprint();
  
  return mtOp<typename T1::pod_type, T1, op_imag>( X );
  }



template<typename T1>
inline
const mtOpCube<typename T1::pod_type, T1, op_imag>
imag(const BaseCube<std::complex<typename T1::pod_type>,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  return mtOpCube<typename T1::pod_type, T1, op_imag>( X.get_ref() );
  }



//
// log

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_log> >::result
log(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_log>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_log>
log(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_log>(A.get_ref());
  }



//
// log2

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_log2> >::result
log2(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_log2>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_log2>
log2(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_log2>(A.get_ref());
  }



//
// log10

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_log10> >::result
log10(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_log10>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_log10>
log10(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_log10>(A.get_ref());
  }



//
// exp

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_exp> >::result
exp(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_exp>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_exp>
exp(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_exp>(A.get_ref());
  }



// exp2

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_exp2> >::result
exp2(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_exp2>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_exp2>
exp2(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_exp2>(A.get_ref());
  }



// exp10

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_exp10> >::result
exp10(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_exp10>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_exp10>
exp10(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_exp10>(A.get_ref());
  }



//
// abs


template<typename T1>
arma_inline
typename enable_if2< (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no), const eOp<T1, eop_abs> >::result
abs(const T1& X)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_abs>(X);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_abs>
abs(const BaseCube<typename T1::elem_type,T1>& X, const typename arma_not_cx<typename T1::elem_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  
  arma_ignore(junk);
  
  return eOpCube<T1, eop_abs>(X.get_ref());
  }



template<typename T1>
inline
typename enable_if2< (is_arma_type<T1>::value && is_cx<typename T1::elem_type>::yes), const mtOp<typename T1::pod_type, T1, op_abs> >::result
abs(const T1& X)
  {
  arma_extra_debug_sigprint();
  
  return mtOp<typename T1::pod_type, T1, op_abs>(X);
  }



template<typename T1>
inline
const mtOpCube<typename T1::pod_type, T1, op_abs>
abs(const BaseCube< std::complex<typename T1::pod_type>,T1>& X, const typename arma_cx_only<typename T1::elem_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  
  arma_ignore(junk);
  
  return mtOpCube<typename T1::pod_type, T1, op_abs>( X.get_ref() );
  }



template<typename T1>
arma_inline
const SpOp<T1, spop_abs>
abs(const SpBase<typename T1::elem_type,T1>& X, const typename arma_not_cx<typename T1::elem_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return SpOp<T1, spop_abs>(X.get_ref());
  }



template<typename T1>
arma_inline
const mtSpOp<typename T1::pod_type, T1, spop_cx_abs>
abs(const SpBase< std::complex<typename T1::pod_type>, T1>& X, const typename arma_cx_only<typename T1::elem_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return mtSpOp<typename T1::pod_type, T1, spop_cx_abs>(X.get_ref());
  }



//
// square

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_square> >::result
square(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_square>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_square>
square(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_square>(A.get_ref());
  }



template<typename T1>
arma_inline
const SpOp<T1, spop_square>
square(const SpBase<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return SpOp<T1, spop_square>(A.get_ref());
  }



//
// sqrt

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_sqrt> >::result
sqrt(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_sqrt>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_sqrt>
sqrt(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_sqrt>(A.get_ref());
  }



template<typename T1>
arma_inline
const SpOp<T1, spop_sqrt>
sqrt(const SpBase<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return SpOp<T1, spop_sqrt>(A.get_ref());
  }



//
// conj

template<typename T1>
arma_inline
const T1&
conj(const Base<typename T1::pod_type,T1>& A)
  {
  arma_extra_debug_sigprint();

  return A.get_ref();
  }



template<typename T1>
arma_inline
const T1&
conj(const BaseCube<typename T1::pod_type,T1>& A)
  {
  arma_extra_debug_sigprint();

  return A.get_ref();
  }



template<typename T1>
arma_inline
const eOp<T1, eop_conj>
conj(const Base<std::complex<typename T1::pod_type>,T1>& A)
  {
  arma_extra_debug_sigprint();

  return eOp<T1, eop_conj>(A.get_ref());
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_conj>
conj(const BaseCube<std::complex<typename T1::pod_type>,T1>& A)
  {
  arma_extra_debug_sigprint();

  return eOpCube<T1, eop_conj>(A.get_ref());
  }



template<typename T1>
arma_inline
const typename Proxy<T1>::stored_type&
conj(const eOp<T1, eop_conj>& A)
  {
  arma_extra_debug_sigprint();
  
  return A.P.Q;
  }



template<typename T1>
arma_inline
const typename ProxyCube<T1>::stored_type&
conj(const eOpCube<T1, eop_conj>& A)
  {
  arma_extra_debug_sigprint();
  
  return A.P.Q;
  }



// pow

template<typename T1>
arma_inline
const eOp<T1, eop_pow>
pow(const Base<typename T1::elem_type,T1>& A, const typename T1::elem_type exponent)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_pow>(A.get_ref(), exponent);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_pow>
pow(const BaseCube<typename T1::elem_type,T1>& A, const typename T1::elem_type exponent)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_pow>(A.get_ref(), exponent);
  }



// pow, specialised handling (non-complex exponent for complex matrices)

template<typename T1>
arma_inline
const eOp<T1, eop_pow>
pow(const Base<typename T1::elem_type,T1>& A, const typename T1::elem_type::value_type exponent)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  return eOp<T1, eop_pow>(A.get_ref(), eT(exponent));
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_pow>
pow(const BaseCube<typename T1::elem_type,T1>& A, const typename T1::elem_type::value_type exponent)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  return eOpCube<T1, eop_pow>(A.get_ref(), eT(exponent));
  }



//
// floor

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_floor> >::result
floor(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_floor>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_floor>
floor(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_floor>(A.get_ref());
  }



//
// ceil

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_ceil> >::result
ceil(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_ceil>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_ceil>
ceil(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_ceil>(A.get_ref());
  }



//
// round

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_round> >::result
round(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_round>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_round>
round(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_round>(A.get_ref());
  }



//
// sign

template<typename T1>
arma_inline
typename enable_if2< is_arma_type<T1>::value, const eOp<T1, eop_sign> >::result
sign(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_sign>(A);
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_sign>
sign(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_sign>(A.get_ref());
  }



//! @}
