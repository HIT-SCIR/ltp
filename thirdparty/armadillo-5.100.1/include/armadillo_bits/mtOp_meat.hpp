// Copyright (C) 2008-2011 Conrad Sanderson
// Copyright (C) 2008-2011 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup mtOp
//! @{



template<typename out_eT, typename T1, typename op_type>
inline
mtOp<out_eT, T1, op_type>::mtOp(const T1& in_m)
  : m(in_m)
  {
  arma_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename op_type>
inline
mtOp<out_eT, T1, op_type>::mtOp(const T1& in_m, const typename T1::elem_type in_aux)
  : m(in_m)
  , aux(in_aux)
  {
  arma_extra_debug_sigprint();
  }
  


template<typename out_eT, typename T1, typename op_type>
inline
mtOp<out_eT, T1, op_type>::mtOp(const T1& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b)
  : m(in_m)
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  {
  arma_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename op_type>
inline
mtOp<out_eT, T1, op_type>::mtOp(const T1& in_m, const typename T1::elem_type in_aux, const uword in_aux_uword_a, const uword in_aux_uword_b)
  : m(in_m)
  , aux(in_aux)
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  {
  arma_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename op_type>
inline
mtOp<out_eT, T1, op_type>::mtOp(const char junk, const T1& in_m, const out_eT in_aux)
  : m(in_m)
  , aux_out_eT(in_aux)
  {
  arma_ignore(junk);
  
  arma_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename op_type>
inline
mtOp<out_eT, T1, op_type>::mtOp(const mtOp_dual_aux_indicator&, const T1& in_m, const typename T1::elem_type in_aux_a, const out_eT in_aux_b)
  : m         (in_m    )
  , aux       (in_aux_a)
  , aux_out_eT(in_aux_b)
  {
  arma_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename op_type>
inline
mtOp<out_eT, T1, op_type>::~mtOp()
  {
  arma_extra_debug_sigprint();
  }



//! @}
