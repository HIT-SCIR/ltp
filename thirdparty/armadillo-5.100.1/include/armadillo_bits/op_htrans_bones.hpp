// Copyright (C) 2008-2013 Conrad Sanderson
// Copyright (C) 2008-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_htrans
//! @{


//! 'hermitian transpose' operation

class op_htrans
  {
  public:
  
  template<typename eT>
  arma_hot arma_inline static void apply_mat_noalias(Mat<eT>& out, const Mat<eT>& A, const typename arma_not_cx<eT>::result* junk = 0);
  
  template<typename eT>
  arma_hot      inline static void apply_mat_noalias(Mat<eT>& out, const Mat<eT>& A, const typename arma_cx_only<eT>::result* junk = 0);
  
  //
  
  template<typename eT>
  arma_hot arma_inline static void apply_mat_inplace(Mat<eT>& out, const typename arma_not_cx<eT>::result* junk = 0);
  
  template<typename eT>
  arma_hot      inline static void apply_mat_inplace(Mat<eT>& out, const typename arma_cx_only<eT>::result* junk = 0);
  
  //
  
  template<typename eT>
  arma_hot arma_inline static void apply_mat(Mat<eT>& out, const Mat<eT>& A, const typename arma_not_cx<eT>::result* junk = 0);
  
  template<typename eT>
  arma_hot      inline static void apply_mat(Mat<eT>& out, const Mat<eT>& A, const typename arma_cx_only<eT>::result* junk = 0);
  
  //
  
  template<typename T1>
  arma_hot inline static void apply_proxy(Mat<typename T1::elem_type>& out, const T1& X);
  
  //
  
  template<typename T1>
  arma_hot inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_htrans>& in, const typename arma_not_cx<typename T1::elem_type>::result* junk = 0);
  
  template<typename T1>
  arma_hot inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_htrans>& in, const typename arma_cx_only<typename T1::elem_type>::result* junk = 0);
  
  //
  
  template<typename T1>
  arma_hot inline static void apply(Mat<typename T1::elem_type>& out, const Op< Op<T1, op_trimat>, op_htrans>& in);
  };



class op_htrans2
  {
  public:
  
  template<typename eT>
  arma_hot inline static void apply_noalias(Mat<eT>& out, const Mat<eT>& A, const eT val);
  
  template<typename eT>
  arma_hot inline static void apply(Mat<eT>& out, const Mat<eT>& A, const eT val);
  
  //
  
  template<typename T1>
  arma_hot inline static void apply_proxy(Mat<typename T1::elem_type>& out, const T1& X, const typename T1::elem_type val);
  
  //
  
  template<typename T1>
  arma_hot inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_htrans2>& in, const typename arma_not_cx<typename T1::elem_type>::result* junk = 0);
  
  template<typename T1>
  arma_hot inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_htrans2>& in, const typename arma_cx_only<typename T1::elem_type>::result* junk = 0);
  };



//! @}
