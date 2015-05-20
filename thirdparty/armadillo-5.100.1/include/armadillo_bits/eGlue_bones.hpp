// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup eGlue
//! @{


template<typename T1, typename T2, typename eglue_type>
class eGlue : public Base<typename T1::elem_type, eGlue<T1, T2, eglue_type> >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef          Proxy<T1>                       proxy1_type;
  typedef          Proxy<T2>                       proxy2_type;
  
  static const bool prefer_at_accessor = (Proxy<T1>::prefer_at_accessor || Proxy<T2>::prefer_at_accessor);
  static const bool has_subview        = (Proxy<T1>::has_subview        || Proxy<T2>::has_subview       );
  static const bool fake_mat           = (Proxy<T1>::fake_mat           || Proxy<T2>::fake_mat          );
  
  static const bool is_col = (Proxy<T1>::is_col || Proxy<T2>::is_col);
  static const bool is_row = (Proxy<T1>::is_row || Proxy<T2>::is_row);
  
  arma_aligned const Proxy<T1> P1;
  arma_aligned const Proxy<T2> P2;
  
  arma_inline ~eGlue();
  arma_inline  eGlue(const T1& in_A, const T2& in_B);
  
  arma_inline uword get_n_rows() const;
  arma_inline uword get_n_cols() const;
  arma_inline uword get_n_elem() const;
  
  arma_inline elem_type operator[] (const uword ii)                   const;
  arma_inline elem_type at         (const uword row, const uword col) const;
  arma_inline elem_type at_alt     (const uword ii)                   const;
  };



//! @}
