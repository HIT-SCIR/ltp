// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup eOp
//! @{



template<typename T1, typename eop_type>
class eOp : public Base<typename T1::elem_type, eOp<T1, eop_type> >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef          Proxy<T1>                       proxy_type;
  
  static const bool prefer_at_accessor = Proxy<T1>::prefer_at_accessor;
  static const bool has_subview        = Proxy<T1>::has_subview;
  static const bool fake_mat           = Proxy<T1>::fake_mat;
  
  static const bool is_row = Proxy<T1>::is_row;
  static const bool is_col = Proxy<T1>::is_col;
  
  arma_aligned const Proxy<T1> P;
  
  arma_aligned       elem_type aux;          //!< storage of auxiliary data, user defined format
  arma_aligned       uword     aux_uword_a;  //!< storage of auxiliary data, uword format
  arma_aligned       uword     aux_uword_b;  //!< storage of auxiliary data, uword format
  
  inline         ~eOp();
  inline explicit eOp(const T1& in_m);
  inline          eOp(const T1& in_m, const elem_type in_aux);
  inline          eOp(const T1& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b);
  inline          eOp(const T1& in_m, const elem_type in_aux, const uword in_aux_uword_a, const uword in_aux_uword_b);
  
  arma_inline uword get_n_rows() const;
  arma_inline uword get_n_cols() const;
  arma_inline uword get_n_elem() const;
  
  arma_inline elem_type operator[] (const uword ii)                   const;
  arma_inline elem_type at         (const uword row, const uword col) const;
  arma_inline elem_type at_alt     (const uword ii)                   const;
  };



//! @}
