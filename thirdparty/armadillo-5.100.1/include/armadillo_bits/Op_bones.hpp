// Copyright (C) 2008-2015 Conrad Sanderson
// Copyright (C) 2008-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup Op
//! @{



//! Class for storing data required for delayed unary operations,
//! such as the operand (e.g. the matrix to which the operation is to be applied) and the unary operator (e.g. inverse).
//! The operand is stored as a reference (which can be optimised away),
//! while the operator is "stored" through the template definition (op_type).
//! The operands can be 'Mat', 'Row', 'Col', 'Op', and 'Glue'.
//! Note that as 'Glue' can be one of the operands, more than one matrix can be stored.
//!
//! For example, we could have:
//! Op< Glue< Mat, Mat, glue_times >, op_htrans >

template<typename T1, typename op_type>
class Op : public Base<typename T1::elem_type, Op<T1, op_type> >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  static const bool is_row = \
     (T1::is_col && (is_same_type<op_type, op_strans>::yes || is_same_type<op_type, op_htrans>::yes || is_same_type<op_type, op_htrans2>::yes))
  || (T1::is_row && (is_same_type<op_type, op_sort>::yes || is_same_type<op_type, op_shuffle>::yes || is_same_type<op_type, op_cumsum_vec>::yes || is_same_type<op_type, op_flipud>::yes || is_same_type<op_type, op_fliplr>::yes))
  || (is_same_type<op_type, op_normalise_rowvec>::yes);
  
  static const bool is_col = \
     (T1::is_row && (is_same_type<op_type, op_strans>::yes || is_same_type<op_type, op_htrans>::yes || is_same_type<op_type, op_htrans2>::yes))
  || (T1::is_col && (is_same_type<op_type, op_sort>::yes || is_same_type<op_type, op_shuffle>::yes || is_same_type<op_type, op_cumsum_vec>::yes || is_same_type<op_type, op_flipud>::yes || is_same_type<op_type, op_fliplr>::yes))
  || (is_same_type<op_type, op_normalise_colvec>::yes)
  || (is_same_type<op_type, op_diagvec>::yes)
  || (is_same_type<op_type, op_vectorise_col>::yes)
  || (is_same_type<op_type, op_nonzeros>::yes);
  
  inline explicit Op(const T1& in_m);
  inline          Op(const T1& in_m, const elem_type in_aux);
  inline          Op(const T1& in_m, const elem_type in_aux,         const uword in_aux_uword_a, const uword in_aux_uword_b);
  inline          Op(const T1& in_m, const uword     in_aux_uword_a, const uword in_aux_uword_b);
  inline          Op(const T1& in_m, const uword     in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c, const char junk);
  inline         ~Op();
    
  
  arma_aligned const T1&       m;            //!< storage of reference to the operand (eg. a matrix)
  arma_aligned       elem_type aux;          //!< storage of auxiliary data, user defined format
  arma_aligned       uword     aux_uword_a;  //!< storage of auxiliary data, uword format
  arma_aligned       uword     aux_uword_b;  //!< storage of auxiliary data, uword format
  arma_aligned       uword     aux_uword_c;  //!< storage of auxiliary data, uword format
  
  };



//! @}
