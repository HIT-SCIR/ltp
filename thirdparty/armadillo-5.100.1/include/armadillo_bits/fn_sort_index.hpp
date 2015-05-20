// Copyright (C) 2009-2015 Conrad Sanderson
// Copyright (C) 2009-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_sort_index
//! @{




//! kept for compatibility with old code
template<typename T1>
inline
const mtOp<uword,T1,op_sort_index>
sort_index
  (
  const Base<typename T1::elem_type,T1>& X,
  const uword sort_type = 0
  )
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (sort_type > 1), "sort_index(): sort_type must be 0 or 1");
  
  return mtOp<uword,T1,op_sort_index>(X.get_ref(), sort_type, uword(0));
  }



//! kept for compatibility with old code
template<typename T1>
inline
const mtOp<uword,T1,op_stable_sort_index>
stable_sort_index
  (
  const Base<typename T1::elem_type,T1>& X,
  const uword sort_type = 0
  )
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (sort_type > 1), "stable_sort_index(): sort_type must be 0 or 1");
  
  return mtOp<uword,T1,op_stable_sort_index>(X.get_ref(), sort_type, uword(0));
  }



template<typename T1, typename T2>
inline
typename
enable_if2
  <
  ( (is_arma_type<T1>::value == true) && (is_same_type<T2, char>::value == true) ),
  const mtOp<uword,T1,op_sort_index>
  >::result
sort_index
  (
  const T1& X,
  const T2* sort_direction
  )
  {
  arma_extra_debug_sigprint();
  
  const char sig = (sort_direction != NULL) ? sort_direction[0] : char(0);
  
  arma_debug_check( ((sig != 'a') && (sig != 'd')), "sort_index(): unknown sort direction" );
  
  return mtOp<uword,T1,op_sort_index>(X, ((sig == 'a') ? uword(0) : uword(1)), uword(0));
  }



template<typename T1, typename T2>
inline
typename
enable_if2
  <
  ( (is_arma_type<T1>::value == true) && (is_same_type<T2, char>::value == true) ),
  const mtOp<uword,T1,op_stable_sort_index>
  >::result
stable_sort_index
  (
  const T1& X,
  const T2* sort_direction
  )
  {
  arma_extra_debug_sigprint();
  
  const char sig = (sort_direction != NULL) ? sort_direction[0] : char(0);
  
  arma_debug_check( ((sig != 'a') && (sig != 'd')), "stable_sort_index(): unknown sort direction" );
  
  return mtOp<uword,T1,op_stable_sort_index>(X, ((sig == 'a') ? uword(0) : uword(1)), uword(0));
  }



//! @}
