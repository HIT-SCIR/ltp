// Copyright (C) 2009-2012 Conrad Sanderson
// Copyright (C) 2009-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_prod
//! @{


//! \brief
//! Delayed product of elements of a matrix along a specified dimension (either rows or columns).
//! The result is stored in a dense matrix that has either one column or one row.
//! For dim = 0, find the sum of each column (i.e. traverse across rows)
//! For dim = 1, find the sum of each row (i.e. traverse across columns)
//! The default is dim = 0.
//! NOTE: this function works differently than in Matlab/Octave.

template<typename T1>
arma_inline
const Op<T1, op_prod>
prod
  (
  const T1& X,
  const uword dim = 0,
  const typename enable_if< is_arma_type<T1>::value       == true  >::result* junk1 = 0,
  const typename enable_if< resolves_to_vector<T1>::value == false >::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return Op<T1, op_prod>(X, dim, 0);
  }



template<typename T1>
arma_inline
const Op<T1, op_prod>
prod
  (
  const T1& X,
  const uword dim,
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return Op<T1, op_prod>(X, dim, 0);
  }



template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
prod
  (
  const T1& X,
  const arma_empty_class junk1 = arma_empty_class(),
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return op_prod::prod( X );
  }



//! \brief
//! Immediate 'product of all values' operation,
//! invoked, for example, by: prod(prod(A))

template<typename T1>
inline
arma_warn_unused
typename T1::elem_type
prod(const Op<T1, op_prod>& in)
  {
  arma_extra_debug_sigprint();
  arma_extra_debug_print("prod(): two consecutive prod() calls detected");
  
  return op_prod::prod( in.m );
  }



template<typename T1>
inline
const Op<Op<T1, op_prod>, op_prod>
prod(const Op<T1, op_prod>& in, const uword dim)
  {
  arma_extra_debug_sigprint();
  
  return Op<Op<T1, op_prod>, op_prod>(in, dim, 0);
  }



template<typename T>
arma_inline
arma_warn_unused
const typename arma_scalar_only<T>::result &
prod(const T& x)
  {
  return x;
  }



//! @}
