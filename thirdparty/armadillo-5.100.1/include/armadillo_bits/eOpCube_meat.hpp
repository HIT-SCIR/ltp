// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup eOpCube
//! @{



template<typename T1, typename eop_type>
eOpCube<T1, eop_type>::eOpCube(const BaseCube<typename T1::elem_type, T1>& in_m)
  : P (in_m.get_ref())
  {
  arma_extra_debug_sigprint();
  }
  


template<typename T1, typename eop_type>
eOpCube<T1, eop_type>::eOpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const typename T1::elem_type in_aux)
  : P   (in_m.get_ref())
  , aux (in_aux)
  {
  arma_extra_debug_sigprint();
  }
  


template<typename T1, typename eop_type>
eOpCube<T1, eop_type>::eOpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b)
  : P           (in_m.get_ref())
  , aux_uword_a (in_aux_uword_a)
  , aux_uword_b (in_aux_uword_b)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename eop_type>
eOpCube<T1, eop_type>::eOpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c)
  : P           (in_m.get_ref())
  , aux_uword_a (in_aux_uword_a)
  , aux_uword_b (in_aux_uword_b)
  , aux_uword_c (in_aux_uword_c)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename eop_type>
eOpCube<T1, eop_type>::eOpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const typename T1::elem_type in_aux, const uword in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c)
  : P           (in_m.get_ref())
  , aux         (in_aux)
  , aux_uword_a (in_aux_uword_a)
  , aux_uword_b (in_aux_uword_b)
  , aux_uword_c (in_aux_uword_c)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename eop_type>
eOpCube<T1, eop_type>::~eOpCube()
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename eop_type>
arma_inline
uword
eOpCube<T1, eop_type>::get_n_rows() const
  {
  return P.get_n_rows();
  }
  


template<typename T1, typename eop_type>
arma_inline
uword
eOpCube<T1, eop_type>::get_n_cols() const
  {
  return P.get_n_cols();
  }



template<typename T1, typename eop_type>
arma_inline
uword
eOpCube<T1, eop_type>::get_n_elem_slice() const
  {
  return P.get_n_elem_slice();
  }



template<typename T1, typename eop_type>
arma_inline
uword
eOpCube<T1, eop_type>::get_n_slices() const
  {
  return P.get_n_slices();
  }



template<typename T1, typename eop_type>
arma_inline
uword
eOpCube<T1, eop_type>::get_n_elem() const
  {
  return P.get_n_elem();
  }



template<typename T1, typename eop_type>
arma_inline
typename T1::elem_type
eOpCube<T1, eop_type>::operator[] (const uword i) const
  {
  return eop_core<eop_type>::process(P[i], aux);
  }



template<typename T1, typename eop_type>
arma_inline
typename T1::elem_type
eOpCube<T1, eop_type>::at(const uword row, const uword col, const uword slice) const
  {
  return eop_core<eop_type>::process(P.at(row, col, slice), aux);
  }



template<typename T1, typename eop_type>
arma_inline
typename T1::elem_type
eOpCube<T1, eop_type>::at_alt(const uword i) const
  {
  return eop_core<eop_type>::process(P.at_alt(i), aux);
  }



//! @}
