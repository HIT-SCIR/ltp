// Copyright (C) 2008-2011 Conrad Sanderson
// Copyright (C) 2008-2011 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup OpCube
//! @{



template<typename T1, typename op_type>
OpCube<T1, op_type>::OpCube(const BaseCube<typename T1::elem_type, T1>& in_m)
  : m(in_m.get_ref())
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename op_type>
OpCube<T1, op_type>::OpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const typename T1::elem_type in_aux)
  : m(in_m.get_ref())
  , aux(in_aux)
  {
  arma_extra_debug_sigprint();
  }
  

template<typename T1, typename op_type>
OpCube<T1, op_type>::OpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const typename T1::elem_type in_aux, const uword in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c)
  : m(in_m.get_ref())
  , aux(in_aux)
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  , aux_uword_c(in_aux_uword_c)
  {
  arma_extra_debug_sigprint();
  }




template<typename T1, typename op_type>
OpCube<T1, op_type>::OpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b)
  : m(in_m.get_ref())
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename op_type>
OpCube<T1, op_type>::OpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c)
  : m(in_m.get_ref())
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  , aux_uword_c(in_aux_uword_c)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename op_type>
OpCube<T1, op_type>::OpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c, const uword in_aux_uword_d, const char)
  : m(in_m.get_ref())
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  , aux_uword_c(in_aux_uword_c)
  , aux_uword_d(in_aux_uword_d)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename op_type>
OpCube<T1, op_type>::~OpCube()
  {
  arma_extra_debug_sigprint();
  }



//! @}
