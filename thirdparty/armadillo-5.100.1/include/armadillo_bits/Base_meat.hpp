// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup Base
//! @{



template<typename elem_type, typename derived>
arma_inline
const derived&
Base<elem_type,derived>::get_ref() const
  {
  return static_cast<const derived&>(*this);
  }



template<typename elem_type, typename derived>
inline
void
Base<elem_type,derived>::print(const std::string extra_text) const
  {
  const Proxy<derived> P( (*this).get_ref() );
  
  const quasi_unwrap< typename Proxy<derived>::stored_type > tmp(P.Q);
  
  tmp.M.impl_print(extra_text);
  }



template<typename elem_type, typename derived>
inline
void
Base<elem_type,derived>::print(std::ostream& user_stream, const std::string extra_text) const
  {
  const Proxy<derived> P( (*this).get_ref() );
  
  const quasi_unwrap< typename Proxy<derived>::stored_type > tmp(P.Q);
  
  tmp.M.impl_print(user_stream, extra_text);
  }
  


template<typename elem_type, typename derived>
inline
void
Base<elem_type,derived>::raw_print(const std::string extra_text) const
  {
  const Proxy<derived> P( (*this).get_ref() );
  
  const quasi_unwrap< typename Proxy<derived>::stored_type > tmp(P.Q);
  
  tmp.M.impl_raw_print(extra_text);
  }



template<typename elem_type, typename derived>
inline
void
Base<elem_type,derived>::raw_print(std::ostream& user_stream, const std::string extra_text) const
  {
  const Proxy<derived> P( (*this).get_ref() );
  
  const quasi_unwrap< typename Proxy<derived>::stored_type > tmp(P.Q);
  
  tmp.M.impl_raw_print(user_stream, extra_text);
  }



template<typename elem_type, typename derived>
inline
arma_warn_unused
elem_type
Base<elem_type,derived>::min() const
  {
  return op_min::min( (*this).get_ref() );
  }



template<typename elem_type, typename derived>
inline
arma_warn_unused
elem_type
Base<elem_type,derived>::max() const
  {
  return op_max::max( (*this).get_ref() );
  }



template<typename elem_type, typename derived>
inline
elem_type
Base<elem_type,derived>::min(uword& index_of_min_val) const
  {
  const Proxy<derived> P( (*this).get_ref() );
  
  return op_min::min_with_index(P, index_of_min_val);
  }



template<typename elem_type, typename derived>
inline
elem_type
Base<elem_type,derived>::max(uword& index_of_max_val) const
  {
  const Proxy<derived> P( (*this).get_ref() );
  
  return op_max::max_with_index(P, index_of_max_val);
  }



template<typename elem_type, typename derived>
inline
elem_type
Base<elem_type,derived>::min(uword& row_of_min_val, uword& col_of_min_val) const
  {
  const Proxy<derived> P( (*this).get_ref() );
  
  uword index;
  
  const elem_type val = op_min::min_with_index(P, index);
  
  const uword local_n_rows = P.get_n_rows();
  
  row_of_min_val = index % local_n_rows;
  col_of_min_val = index / local_n_rows;
  
  return val;
  }



template<typename elem_type, typename derived>
inline
elem_type
Base<elem_type,derived>::max(uword& row_of_max_val, uword& col_of_max_val) const
  {
  const Proxy<derived> P( (*this).get_ref() );
  
  uword index;
  
  const elem_type val = op_max::max_with_index(P, index);
  
  const uword local_n_rows = P.get_n_rows();
  
  row_of_max_val = index % local_n_rows;
  col_of_max_val = index / local_n_rows;
  
  return val;
  }



//
// extra functions defined in Base_inv_yes

template<typename derived>
arma_inline
const Op<derived,op_inv>
Base_inv_yes<derived>::i(const bool slow) const
  {
  return Op<derived,op_inv>( static_cast<const derived&>(*this), ((slow == false) ? 0 : 1), 0 );
  }



template<typename derived>
arma_inline
const Op<derived,op_inv>
Base_inv_yes<derived>::i(const char* method) const
  {
  const char sig = (method != NULL) ? method[0] : char(0);
  
  arma_debug_check( ((sig != 's') && (sig != 'f')), "Base::i(): unknown method specified" );
  
  return Op<derived,op_inv>( static_cast<const derived&>(*this), ((sig == 'f') ? 0 : 1), 0 );
  }



//
// extra functions defined in Base_eval_Mat

template<typename elem_type, typename derived>
arma_inline
const derived&
Base_eval_Mat<elem_type, derived>::eval() const
  {
  arma_extra_debug_sigprint();
  
  return static_cast<const derived&>(*this);
  }



//
// extra functions defined in Base_eval_expr

template<typename elem_type, typename derived>
arma_inline
Mat<elem_type>
Base_eval_expr<elem_type, derived>::eval() const
  {
  arma_extra_debug_sigprint();
  
  return Mat<elem_type>( static_cast<const derived&>(*this) );
  }



//
// extra functions defined in Base_trans_cx

template<typename derived>
arma_inline
const Op<derived,op_htrans>
Base_trans_cx<derived>::t() const
  {
  return Op<derived,op_htrans>( static_cast<const derived&>(*this) );
  }



template<typename derived>
arma_inline
const Op<derived,op_htrans>
Base_trans_cx<derived>::ht() const
  {
  return Op<derived,op_htrans>( static_cast<const derived&>(*this) );
  }



template<typename derived>
arma_inline
const Op<derived,op_strans>
Base_trans_cx<derived>::st() const
  {
  return Op<derived,op_strans>( static_cast<const derived&>(*this) );
  }



//
// extra functions defined in Base_trans_default

template<typename derived>
arma_inline
const Op<derived,op_htrans>
Base_trans_default<derived>::t() const
  {
  return Op<derived,op_htrans>( static_cast<const derived&>(*this) );
  }



template<typename derived>
arma_inline
const Op<derived,op_htrans>
Base_trans_default<derived>::ht() const
  {
  return Op<derived,op_htrans>( static_cast<const derived&>(*this) );
  }



template<typename derived>
arma_inline
const Op<derived,op_htrans>
Base_trans_default<derived>::st() const
  {
  return Op<derived,op_htrans>( static_cast<const derived&>(*this) );
  }



//! @}
