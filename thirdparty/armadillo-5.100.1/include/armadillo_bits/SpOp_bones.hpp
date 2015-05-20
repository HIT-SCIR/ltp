// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup SpOp
//! @{



template<typename T1, typename op_type>
class SpOp : public SpBase<typename T1::elem_type, SpOp<T1, op_type> >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  static const bool is_row = (T1::is_row && is_spop_elem<op_type>::value) || ( T1::is_col && (is_same_type<op_type, spop_strans>::value || is_same_type<op_type, spop_htrans>::value) );
  static const bool is_col = (T1::is_col && is_spop_elem<op_type>::value) || ( T1::is_row && (is_same_type<op_type, spop_strans>::value || is_same_type<op_type, spop_htrans>::value) );
  
  inline explicit SpOp(const T1& in_m);
  inline          SpOp(const T1& in_m, const elem_type in_aux);
  inline          SpOp(const T1& in_m, const uword     in_aux_uword_a, const uword in_aux_uword_b);
  inline         ~SpOp();
  
  
  arma_aligned const T1&       m;            //!< storage of reference to the operand (eg. a matrix)
  arma_aligned       elem_type aux;          //!< storage of auxiliary data, user defined format
  arma_aligned       uword     aux_uword_a;  //!< storage of auxiliary data, uword format
  arma_aligned       uword     aux_uword_b;  //!< storage of auxiliary data, uword format
  };



//! @}
