// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup SpOp
//! @{



template<typename T1, typename op_type>
inline
SpOp<T1, op_type>::SpOp(const T1& in_m)
  : m(in_m)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename op_type>
inline
SpOp<T1, op_type>::SpOp(const T1& in_m, const typename T1::elem_type in_aux)
  : m(in_m)
  , aux(in_aux)
  {
  arma_extra_debug_sigprint();
  }
  


template<typename T1, typename op_type>
inline
SpOp<T1, op_type>::SpOp(const T1& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b)
  : m(in_m)
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename op_type>
inline
SpOp<T1, op_type>::~SpOp()
  {
  arma_extra_debug_sigprint();
  }



//! @}
