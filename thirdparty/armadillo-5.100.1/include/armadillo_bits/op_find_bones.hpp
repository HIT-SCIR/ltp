// Copyright (C) 2010-2014 Conrad Sanderson
// Copyright (C) 2010-2014 NICTA (www.nicta.com.au)
// Copyright (C) 2010 Dimitrios Bouzas
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup op_find
//! @{



class op_find
  {
  public:
  
  template<typename T1>
  inline static uword
  helper
    (
    Mat<uword>& indices,
    const Base<typename T1::elem_type, T1>& X
    );
  
  template<typename T1, typename op_type>
  inline static uword
  helper
    (
    Mat<uword>& indices,
    const mtOp<uword, T1, op_type>& X,
    const typename arma_op_rel_only<op_type>::result junk1 = 0,
    const typename arma_not_cx<typename T1::elem_type>::result junk2 = 0
    );
  
  template<typename T1, typename op_type>
  inline static uword
  helper
    (
    Mat<uword>& indices,
    const mtOp<uword, T1, op_type>& X,
    const typename arma_op_rel_only<op_type>::result junk1 = 0,
    const typename arma_cx_only<typename T1::elem_type>::result junk2 = 0
    );
  
  template<typename T1, typename T2, typename glue_type>
  inline static uword
  helper
    (
    Mat<uword>& indices,
    const mtGlue<uword, T1, T2, glue_type>& X,
    const typename arma_glue_rel_only<glue_type>::result junk1 = 0,
    const typename arma_not_cx<typename T1::elem_type>::result junk2 = 0,
    const typename arma_not_cx<typename T2::elem_type>::result junk3 = 0
    );
  
  template<typename T1, typename T2, typename glue_type>
  inline static uword
  helper
    (
    Mat<uword>& indices,
    const mtGlue<uword, T1, T2, glue_type>& X,
    const typename arma_glue_rel_only<glue_type>::result junk1 = 0,
    const typename arma_cx_only<typename T1::elem_type>::result junk2 = 0,
    const typename arma_cx_only<typename T2::elem_type>::result junk3 = 0
    );
  
  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, op_find>& X);
  };



class op_find_simple
  {
  public:
  
  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, op_find_simple>& X);
  };



class op_find_finite
  {
  public:
  
  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, op_find_finite>& X);
  };



class op_find_nonfinite
  {
  public:
  
  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, op_find_nonfinite>& X);
  };



//! @}
