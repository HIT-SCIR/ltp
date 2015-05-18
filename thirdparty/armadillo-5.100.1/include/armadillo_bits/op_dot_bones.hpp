// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_dot
//! @{

//! \brief
//! dot product operation 

class op_dot
  {
  public:
  
  template<typename eT>
  arma_hot arma_pure arma_inline static
  typename arma_not_cx<eT>::result
  direct_dot_arma(const uword n_elem, const eT* const A, const eT* const B);
  
  template<typename eT>
  arma_hot arma_pure inline static
  typename arma_cx_only<eT>::result
  direct_dot_arma(const uword n_elem, const eT* const A, const eT* const B);
  
  template<typename eT>
  arma_hot arma_pure inline static typename arma_real_only<eT>::result
  direct_dot(const uword n_elem, const eT* const A, const eT* const B);
  
  template<typename eT>
  arma_hot arma_pure inline static typename arma_cx_only<eT>::result
  direct_dot(const uword n_elem, const eT* const A, const eT* const B);
  
  template<typename eT>
  arma_hot arma_pure inline static typename arma_integral_only<eT>::result
  direct_dot(const uword n_elem, const eT* const A, const eT* const B);
  
  
  template<typename eT>
  arma_hot arma_pure inline static eT direct_dot(const uword n_elem, const eT* const A, const eT* const B, const eT* C);
  
  template<typename T1, typename T2>
  arma_hot inline static typename T1::elem_type apply(const T1& X, const T2& Y);
  
  template<typename T1, typename T2>
  arma_hot inline static typename  arma_not_cx<typename T1::elem_type>::result apply_proxy(const Proxy<T1>& PA, const Proxy<T2>& PB);
  
  template<typename T1, typename T2>
  arma_hot inline static typename arma_cx_only<typename T1::elem_type>::result apply_proxy(const Proxy<T1>& PA, const Proxy<T2>& PB);
  };



//! \brief
//! normalised dot product operation 

class op_norm_dot
  {
  public:
  
  template<typename T1, typename T2>
  arma_hot inline static typename T1::elem_type apply(const T1& X, const T2& Y);
  };



//! \brief
//! complex conjugate dot product operation

class op_cdot
  {
  public:
  
  template<typename eT>
  arma_hot inline static eT direct_cdot_arma(const uword n_elem, const eT* const A, const eT* const B);
  
  template<typename eT>
  arma_hot inline static eT direct_cdot(const uword n_elem, const eT* const A, const eT* const B);
  
  template<typename T1, typename T2>
  arma_hot inline static typename T1::elem_type apply       (const T1& X, const T2& Y);
  
  template<typename T1, typename T2>
  arma_hot inline static typename T1::elem_type apply_unwrap(const T1& X, const T2& Y);
  
  template<typename T1, typename T2>
  arma_hot inline static typename T1::elem_type apply_proxy (const T1& X, const T2& Y);
  };



class op_dot_mixed
  {
  public:
  
  template<typename T1, typename T2>
  arma_hot inline static
  typename promote_type<typename T1::elem_type, typename T2::elem_type>::result
  apply(const T1& A, const T2& B);
  };



//! @}
