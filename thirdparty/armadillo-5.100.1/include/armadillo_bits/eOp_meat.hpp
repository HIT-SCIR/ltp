// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup eOp
//! @{



template<typename T1, typename eop_type>
eOp<T1, eop_type>::eOp(const T1& in_m)
  : P(in_m)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename eop_type>
eOp<T1, eop_type>::eOp(const T1& in_m, const typename T1::elem_type in_aux)
  : P(in_m)
  , aux(in_aux)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename eop_type>
eOp<T1, eop_type>::eOp(const T1& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b)
  : P(in_m)
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename eop_type>
eOp<T1, eop_type>::eOp(const T1& in_m, const typename T1::elem_type in_aux, const uword in_aux_uword_a, const uword in_aux_uword_b)
  : P(in_m)
  , aux(in_aux)
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename eop_type>
eOp<T1, eop_type>::~eOp()
  {
  arma_extra_debug_sigprint();
  }

  

template<typename T1, typename eop_type>
arma_inline
uword
eOp<T1, eop_type>::get_n_rows() const
  {
  return is_row ? 1 : P.get_n_rows();
  }
  


template<typename T1, typename eop_type>
arma_inline
uword
eOp<T1, eop_type>::get_n_cols() const
  {
  return is_col ? 1 : P.get_n_cols();
  }



template<typename T1, typename eop_type>
arma_inline
uword
eOp<T1, eop_type>::get_n_elem() const
  {
  return P.get_n_elem();
  }



template<typename T1, typename eop_type>
arma_inline
typename T1::elem_type
eOp<T1, eop_type>::operator[] (const uword ii) const
  {
  return eop_core<eop_type>::process(P[ii], aux);
  }



template<typename T1, typename eop_type>
arma_inline
typename T1::elem_type
eOp<T1, eop_type>::at(const uword row, const uword col) const
  {
  if(is_row)
    {
    return eop_core<eop_type>::process(P.at(0, col), aux);
    }
  else
  if(is_col)
    {
    return eop_core<eop_type>::process(P.at(row, 0), aux);
    }
  else
    {
    return eop_core<eop_type>::process(P.at(row, col), aux);
    }
  }



template<typename T1, typename eop_type>
arma_inline
typename T1::elem_type
eOp<T1, eop_type>::at_alt(const uword ii) const
  {
  return eop_core<eop_type>::process(P.at_alt(ii), aux);
  }



//! @}
