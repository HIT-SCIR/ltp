// Copyright (C) 2012-2014 Conrad Sanderson
// Copyright (C) 2012-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup SpBase
//! @{



template<typename elem_type, typename derived>
struct SpBase_eval_SpMat
  {
  inline const derived& eval() const;
  };


template<typename elem_type, typename derived>
struct SpBase_eval_expr
  {
  inline SpMat<elem_type> eval() const;   //!< force the immediate evaluation of a delayed expression
  };


template<typename elem_type, typename derived, bool condition>
struct SpBase_eval {};

template<typename elem_type, typename derived>
struct SpBase_eval<elem_type, derived, true>  { typedef SpBase_eval_SpMat<elem_type, derived> result; };

template<typename elem_type, typename derived>
struct SpBase_eval<elem_type, derived, false> { typedef SpBase_eval_expr<elem_type, derived>  result; };



template<typename elem_type, typename derived>
struct SpBase
  : public SpBase_eval<elem_type, derived, is_SpMat<derived>::value>::result
  {
  arma_inline const derived& get_ref() const;
  
  inline const SpOp<derived,spop_htrans>  t() const;  //!< Hermitian transpose
  inline const SpOp<derived,spop_htrans> ht() const;  //!< Hermitian transpose
  inline const SpOp<derived,spop_strans> st() const;  //!< simple transpose
  
  inline void print(const std::string extra_text = "") const;
  inline void print(std::ostream& user_stream, const std::string extra_text = "") const;
  
  inline void raw_print(const std::string extra_text = "") const;
  inline void raw_print(std::ostream& user_stream, const std::string extra_text = "") const;
  
  inline void print_dense(const std::string extra_text = "") const;
  inline void print_dense(std::ostream& user_stream, const std::string extra_text = "") const;
  
  inline void raw_print_dense(const std::string extra_text = "") const;
  inline void raw_print_dense(std::ostream& user_stream, const std::string extra_text = "") const;
  
  inline arma_warn_unused elem_type min() const;
  inline arma_warn_unused elem_type max() const;
  
  inline elem_type min(uword& index_of_min_val) const;
  inline elem_type max(uword& index_of_max_val) const;
  
  inline elem_type min(uword& row_of_min_val, uword& col_of_min_val) const;
  inline elem_type max(uword& row_of_max_val, uword& col_of_max_val) const;
  };



//! @}
