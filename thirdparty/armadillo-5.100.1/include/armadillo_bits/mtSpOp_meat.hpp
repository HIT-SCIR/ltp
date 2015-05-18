// Copyright (C) 2012 Ryan Curtin
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup mtSpOp
//! @{



template<typename out_eT, typename T1, typename op_type>
inline
mtSpOp<out_eT, T1, op_type>::mtSpOp(const T1& in_m)
  : m(in_m)
  {
  arma_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename op_type>
inline
mtSpOp<out_eT, T1, op_type>::mtSpOp(const T1& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b)
  : m(in_m)
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  {
  arma_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename op_type>
inline
mtSpOp<out_eT, T1, op_type>::~mtSpOp()
  {
  arma_extra_debug_sigprint();
  }
