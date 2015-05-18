// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2011 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_reshape
//! @{



template<typename T1>
inline
typename enable_if2< is_arma_type<T1>::value, const Op<T1, op_reshape> >::result
reshape(const T1& X, const uword in_n_rows, const uword in_n_cols)
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_reshape>(X, in_n_rows, in_n_cols);
  }



template<typename T1>
inline
const Op<T1, op_reshape_ext>
reshape(const Base<typename T1::elem_type,T1>& X, const uword in_n_rows, const uword in_n_cols, const uword dim)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (dim > 1), "reshape(): dim must be 0 or 1");
  
  return Op<T1, op_reshape_ext>(X.get_ref(), in_n_rows, in_n_cols, dim, 'j');
  }



template<typename T1>
inline
const OpCube<T1, op_reshape_ext>
reshape(const BaseCube<typename T1::elem_type,T1>& X, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices, const uword dim = 0)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (dim > 1), "reshape(): dim must be 0 or 1");
  
  return OpCube<T1, op_reshape_ext>(X.get_ref(), in_n_rows, in_n_cols, in_n_slices, dim, 'j');
  }



//! @}
