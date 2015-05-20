// Copyright (C) 2010-2015 Conrad Sanderson
// Copyright (C) 2010-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup eglue_core
//! @{



template<typename eglue_type>
struct eglue_core
  {
  
  // matrices
  
  template<typename outT, typename T1, typename T2> arma_hot inline static void apply(outT& out, const eGlue<T1, T2, eglue_type>& x);
  
  template<typename T1, typename T2> arma_hot inline static void apply_inplace_plus (Mat<typename T1::elem_type>& out, const eGlue<T1, T2, eglue_type>& x);
  template<typename T1, typename T2> arma_hot inline static void apply_inplace_minus(Mat<typename T1::elem_type>& out, const eGlue<T1, T2, eglue_type>& x);
  template<typename T1, typename T2> arma_hot inline static void apply_inplace_schur(Mat<typename T1::elem_type>& out, const eGlue<T1, T2, eglue_type>& x);
  template<typename T1, typename T2> arma_hot inline static void apply_inplace_div  (Mat<typename T1::elem_type>& out, const eGlue<T1, T2, eglue_type>& x);
  
  
  // cubes
  
  template<typename T1, typename T2> arma_hot inline static void apply(Cube<typename T1::elem_type>& out, const eGlueCube<T1, T2, eglue_type>& x);
  
  template<typename T1, typename T2> arma_hot inline static void apply_inplace_plus (Cube<typename T1::elem_type>& out, const eGlueCube<T1, T2, eglue_type>& x);
  template<typename T1, typename T2> arma_hot inline static void apply_inplace_minus(Cube<typename T1::elem_type>& out, const eGlueCube<T1, T2, eglue_type>& x);
  template<typename T1, typename T2> arma_hot inline static void apply_inplace_schur(Cube<typename T1::elem_type>& out, const eGlueCube<T1, T2, eglue_type>& x);
  template<typename T1, typename T2> arma_hot inline static void apply_inplace_div  (Cube<typename T1::elem_type>& out, const eGlueCube<T1, T2, eglue_type>& x);
  };



class eglue_plus : public eglue_core<eglue_plus>
  {
  public:
  
  inline static const char* text() { return "addition"; }
  };



class eglue_minus : public eglue_core<eglue_minus>
  {
  public:
  
  inline static const char* text() { return "subtraction"; }
  };



class eglue_div : public eglue_core<eglue_div>
  {
  public:
  
  inline static const char* text() { return "element-wise division"; }
  };



class eglue_schur : public eglue_core<eglue_schur>
  {
  public:
  
  inline static const char* text() { return "element-wise multiplication"; }
  };



//! @}
