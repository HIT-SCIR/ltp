// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_sort
//! @{


//! kept for compatibility with old code
template<typename T1>
arma_inline
typename
enable_if2
  <
  ( (is_arma_type<T1>::value == true) && (resolves_to_vector<T1>::value == false) ),
  const Op<T1, op_sort>
  >::result
sort
  (
  const T1&   X,
  const uword sort_type = 0,
  const uword dim       = 0
  )
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_sort>(X, sort_type, dim);
  }



template<typename T1, typename T2>
arma_inline
typename
enable_if2
  <
  ( (is_arma_type<T1>::value == true) && (resolves_to_vector<T1>::value == false) && (is_same_type<T2, char>::value) ),
  const Op<T1, op_sort>
  >::result
sort
  (
  const T1&   X,
  const T2*   sort_direction,
  const uword dim           = 0
  )
  {
  arma_extra_debug_sigprint();
  
  const char sig = (sort_direction != NULL) ? sort_direction[0] : char(0);
  
  arma_debug_check( (sig != 'a') && (sig != 'd'), "sort(): unknown sort direction");
  
  const uword sort_type = (sig == 'a') ? 0 : 1;
  
  return Op<T1, op_sort>(X, sort_type, dim);
  }



//! kept for compatibility with old code
template<typename T1>
arma_inline
typename
enable_if2
  <
  ( (is_arma_type<T1>::value == true) && (resolves_to_vector<T1>::value == true) ),
  const Op<T1, op_sort>
  >::result
sort
  (
  const T1&   X,
  const uword sort_type = 0
  )
  {
  arma_extra_debug_sigprint();
  
  const uword dim = (T1::is_col) ? 0 : 1;
  
  return Op<T1, op_sort>(X, sort_type, dim);
  }



template<typename T1, typename T2>
arma_inline
typename
enable_if2
  <
  ( (is_arma_type<T1>::value == true) && (resolves_to_vector<T1>::value == true) && (is_same_type<T2, char>::value) ),
  const Op<T1, op_sort>
  >::result
sort
  (
  const T1& X,
  const T2* sort_direction
  )
  {
  arma_extra_debug_sigprint();
  
  const char sig = (sort_direction != NULL) ? sort_direction[0] : char(0);
  
  arma_debug_check( (sig != 'a') && (sig != 'd'), "sort(): unknown sort direction");
  
  const uword sort_type = (sig == 'a') ? 0 : 1;
  const uword dim       = (T1::is_col) ? 0 : 1;
  
  return Op<T1, op_sort>(X, sort_type, dim);
  }



//! @}
