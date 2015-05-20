// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup eGlueCube
//! @{


template<typename T1, typename T2, typename eglue_type>
class eGlueCube : public BaseCube<typename T1::elem_type, eGlueCube<T1, T2, eglue_type> >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  static const bool prefer_at_accessor = (ProxyCube<T1>::prefer_at_accessor || ProxyCube<T2>::prefer_at_accessor);
  static const bool has_subview        = (ProxyCube<T1>::has_subview        || ProxyCube<T2>::has_subview       );
  
  arma_aligned const ProxyCube<T1> P1;
  arma_aligned const ProxyCube<T2> P2;
  
  arma_inline ~eGlueCube();
  arma_inline  eGlueCube(const T1& in_A, const T2& in_B);
  
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
