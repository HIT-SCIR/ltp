// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup eOpCube
//! @{



template<typename T1, typename eop_type>
class eOpCube : public BaseCube<typename T1::elem_type, eOpCube<T1, eop_type> >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  static const bool prefer_at_accessor = ProxyCube<T1>::prefer_at_accessor;
  static const bool has_subview        = ProxyCube<T1>::has_subview;
  
  arma_aligned const ProxyCube<T1> P;
  arma_aligned       elem_type     aux;          //!< storage of auxiliary data, user defined format
  arma_aligned       uword         aux_uword_a;  //!< storage of auxiliary data, uword format
  arma_aligned       uword         aux_uword_b;  //!< storage of auxiliary data, uword format
  arma_aligned       uword         aux_uword_c;  //!< storage of auxiliary data, uword format
  
  inline         ~eOpCube();
  inline explicit eOpCube(const BaseCube<typename T1::elem_type, T1>& in_m);
  inline          eOpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const elem_type in_aux);
  inline          eOpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b);
  inline          eOpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c);
  inline          eOpCube(const BaseCube<typename T1::elem_type, T1>& in_m, const elem_type in_aux, const uword in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c);
  
  arma_inline uword get_n_rows()       const;
  arma_inline uword get_n_cols()       const;
  arma_inline uword get_n_elem_slice() const;
  arma_inline uword get_n_slices()     const;
  arma_inline uword get_n_elem()       const;
  
  arma_inline elem_type operator[] (const uword i)                                       const;
  arma_inline elem_type at         (const uword row, const uword col, const uword slice) const;
  arma_inline elem_type at_alt     (const uword i)                                       const;
  };



//! @}
