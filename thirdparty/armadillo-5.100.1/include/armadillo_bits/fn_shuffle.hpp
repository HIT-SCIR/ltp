// Copyright (C) 2009-2013 Conrad Sanderson
// Copyright (C) 2009-2013 NICTA (www.nicta.com.au)
// Copyright (C) 2009-2010 Dimitrios Bouzas
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup fn_shuffle
//! @{

//! \brief
//! Shuffle the rows or the columns of a matrix or vector in random fashion.
//! If dim = 0, shuffle the columns (default operation).
//! If dim = 1, shuffle the rows.

template<typename T1>
arma_inline
const Op<T1, op_shuffle>
shuffle
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
  
  return Op<T1, op_shuffle>(X, dim, 0);
  }



template<typename T1>
arma_inline
const Op<T1, op_shuffle>
shuffle
  (
  const T1& X,
  const uword dim,
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return Op<T1, op_shuffle>(X, dim, 0);
  }



template<typename T1>
arma_inline
const Op<T1, op_shuffle>
shuffle
  (
  const T1& X,
  const arma_empty_class junk1 = arma_empty_class(),
  const typename enable_if<resolves_to_vector<T1>::value == true>::result* junk2 = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  return T1::is_row ? Op<T1, op_shuffle>(X, 1, 0) : Op<T1, op_shuffle>(X, 0, 0);
  }



//! @}
