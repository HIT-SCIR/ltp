// Copyright (C) 2011-2013 Conrad Sanderson
// Copyright (C) 2011-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup GenCube
//! @{


//! support class for generator functions (eg. zeros, randu, randn, ...)
template<typename eT, typename gen_type>
class GenCube : public BaseCube<eT, GenCube<eT, gen_type> >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  static const bool prefer_at_accessor = false;
  static const bool is_simple          = (is_same_type<gen_type, gen_ones_full>::value) || (is_same_type<gen_type, gen_zeros>::value); 
  
  arma_aligned const uword n_rows;
  arma_aligned const uword n_cols;
  arma_aligned const uword n_slices;
  
  arma_inline  GenCube(const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  arma_inline ~GenCube();
  
  arma_inline static eT generate();
  
  arma_inline eT operator[] (const uword i)                                       const;
  arma_inline eT at         (const uword row, const uword col, const uword slice) const;
  arma_inline eT at_alt     (const uword i)                                       const;
  
  inline void apply              (Cube<eT>& out) const;
  inline void apply_inplace_plus (Cube<eT>& out) const;
  inline void apply_inplace_minus(Cube<eT>& out) const;
  inline void apply_inplace_schur(Cube<eT>& out) const;
  inline void apply_inplace_div  (Cube<eT>& out) const;
  };



//! @}
