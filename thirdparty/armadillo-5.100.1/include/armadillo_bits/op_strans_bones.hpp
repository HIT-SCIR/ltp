// Copyright (C) 2008-2013 Conrad Sanderson
// Copyright (C) 2008-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_strans
//! @{


//! 'matrix transpose' operation (simple transpose, ie. without taking the conjugate of the elements)

class op_strans
  {
  public:
  
  template<const bool do_flip, const uword row, const uword col>
  struct pos
    {
    static const uword n2 = (do_flip == false) ? (row + col*2) : (col + row*2);
    static const uword n3 = (do_flip == false) ? (row + col*3) : (col + row*3);
    static const uword n4 = (do_flip == false) ? (row + col*4) : (col + row*4);
    };
  
  template<typename eT, typename TA>
  arma_hot inline static void apply_mat_noalias_tinysq(Mat<eT>& out, const TA& A);
  
  template<typename eT, typename TA>
  arma_hot inline static void apply_mat_noalias(Mat<eT>& out, const TA& A);
  
  template<typename eT>
  arma_hot inline static void apply_mat_inplace(Mat<eT>& out);
  
  template<typename eT, typename TA>
  arma_hot inline static void apply_mat(Mat<eT>& out, const TA& A);
  
  template<typename T1>
  arma_hot inline static void apply_proxy(Mat<typename T1::elem_type>& out, const T1& X);
  
  template<typename T1>
  arma_hot inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_strans>& in);
  };



class op_strans2
  {
  public:
  
  template<const bool do_flip, const uword row, const uword col>
  struct pos
    {
    static const uword n2 = (do_flip == false) ? (row + col*2) : (col + row*2);
    static const uword n3 = (do_flip == false) ? (row + col*3) : (col + row*3);
    static const uword n4 = (do_flip == false) ? (row + col*4) : (col + row*4);
    };
  
  template<typename eT, typename TA>
  arma_hot inline static void apply_noalias_tinysq(Mat<eT>& out, const TA& A, const eT val);
  
  template<typename eT, typename TA>
  arma_hot inline static void apply_noalias(Mat<eT>& out, const TA& A, const eT val);
  
  template<typename eT, typename TA>
  arma_hot inline static void apply(Mat<eT>& out, const TA& A, const eT val);
  
  template<typename T1>
  arma_hot inline static void apply_proxy(Mat<typename T1::elem_type>& out, const T1& X, const typename T1::elem_type val);
  
  // NOTE: there is no direct handling of Op<T1,op_strans2>, as op_strans2::apply_proxy() is currently only called by op_htrans2 for non-complex numbers
  };



//! @}
