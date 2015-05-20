// Copyright (C) 2014 Conrad Sanderson
// Copyright (C) 2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_normalise
//! @{



template<typename T1>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && resolves_to_colvector<T1>::value),
  const Op<T1, op_normalise_colvec>
  >::result
normalise
  (
  const T1&   X,
  const uword p = uword(2),
  const arma_empty_class junk1 = arma_empty_class(),
  const typename arma_real_or_cx_only<typename T1::elem_type>::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return Op<T1, op_normalise_colvec>(X, p, 0);
  }



template<typename T1>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && resolves_to_rowvector<T1>::value),
  const Op<T1, op_normalise_rowvec>
  >::result
normalise
  (
  const T1&   X,
  const uword p = uword(2),
  const arma_empty_class junk1 = arma_empty_class(),
  const typename arma_real_or_cx_only<typename T1::elem_type>::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return Op<T1, op_normalise_rowvec>(X, p, 0);
  }



template<typename T1>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && (resolves_to_vector<T1>::value == false)),
  const Op<T1, op_normalise_mat>
  >::result
normalise
  (
  const T1&   X,
  const uword p = uword(2),
  const uword dim = 0,
  const typename arma_real_or_cx_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return Op<T1, op_normalise_mat>(X, p, dim);
  }



//! @}
