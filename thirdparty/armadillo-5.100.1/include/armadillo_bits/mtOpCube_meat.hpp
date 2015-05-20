// Copyright (C) 2008-2011 Conrad Sanderson
// Copyright (C) 2008-2011 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup mtOpCube
//! @{



template<typename out_eT, typename T1, typename op_type>
inline
mtOpCube<out_eT, T1, op_type>::mtOpCube(const T1& in_m)
  : m(in_m)
  {
  arma_extra_debug_sigprint();
  }
  


template<typename out_eT, typename T1, typename op_type>
inline
mtOpCube<out_eT, T1, op_type>::mtOpCube(const T1& in_m, const typename T1::elem_type in_aux)
  : m(in_m)
  , aux(in_aux)
  {
  arma_extra_debug_sigprint();
  }
  


template<typename out_eT, typename T1, typename op_type>
inline
mtOpCube<out_eT, T1, op_type>::mtOpCube(const T1& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c)
  : m(in_m)
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  , aux_uword_c(in_aux_uword_c)
  {
  arma_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename op_type>
inline
mtOpCube<out_eT, T1, op_type>::mtOpCube(const T1& in_m, const typename T1::elem_type in_aux, const uword in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c)
  : m(in_m)
  , aux(in_aux)
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  , aux_uword_c(in_aux_uword_c)
  {
  arma_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename op_type>
inline
mtOpCube<out_eT, T1, op_type>::mtOpCube(const char junk, const T1& in_m, const out_eT in_aux)
  : m(in_m)
  , aux_out_eT(in_aux)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  }



template<typename out_eT, typename T1, typename op_type>
inline
mtOpCube<out_eT, T1, op_type>::~mtOpCube()
  {
  arma_extra_debug_sigprint();
  }



//! @}
