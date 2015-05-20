// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup BaseCube
//! @{



template<typename elem_type, typename derived>
arma_inline
const derived&
BaseCube<elem_type,derived>::get_ref() const
  {
  return static_cast<const derived&>(*this);
  }



template<typename elem_type, typename derived>
inline
void
BaseCube<elem_type,derived>::print(const std::string extra_text) const
  {
  const unwrap_cube<derived> tmp( (*this).get_ref() );
  
  tmp.M.impl_print(extra_text);
  }



template<typename elem_type, typename derived>
inline
void
BaseCube<elem_type,derived>::print(std::ostream& user_stream, const std::string extra_text) const
  {
  const unwrap_cube<derived> tmp( (*this).get_ref() );
  
  tmp.M.impl_print(user_stream, extra_text);
  }
  


template<typename elem_type, typename derived>
inline
void
BaseCube<elem_type,derived>::raw_print(const std::string extra_text) const
  {
  const unwrap_cube<derived> tmp( (*this).get_ref() );
  
  tmp.M.impl_raw_print(extra_text);
  }



template<typename elem_type, typename derived>
inline
void
BaseCube<elem_type,derived>::raw_print(std::ostream& user_stream, const std::string extra_text) const
  {
  const unwrap_cube<derived> tmp( (*this).get_ref() );
  
  tmp.M.impl_raw_print(user_stream, extra_text);
  }
  


//
// extra functions defined in BaseCube_eval_Cube

template<typename elem_type, typename derived>
arma_inline
const derived&
BaseCube_eval_Cube<elem_type, derived>::eval() const
  {
  arma_extra_debug_sigprint();
  
  return static_cast<const derived&>(*this);
  }



//
// extra functions defined in BaseCube_eval_expr

template<typename elem_type, typename derived>
arma_inline
Cube<elem_type>
BaseCube_eval_expr<elem_type, derived>::eval() const
  {
  arma_extra_debug_sigprint();
  
  return Cube<elem_type>( static_cast<const derived&>(*this) );
  }



//! @}
