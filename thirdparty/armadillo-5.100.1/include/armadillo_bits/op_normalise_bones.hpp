// Copyright (C) 2014 Conrad Sanderson
// Copyright (C) 2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup op_normalise
//! @{



class op_normalise_colvec
  {
  public:
  
  template<typename T1> inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_normalise_colvec>& in);
  };



class op_normalise_rowvec
  {
  public:
  
  template<typename T1> inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_normalise_rowvec>& in);
  };



class op_normalise_mat
  {
  public:
  
  template<typename T1> inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_normalise_mat>& in);
  
  template<typename eT> inline static void apply(Mat<eT>& out, const Mat<eT>& A, const uword p, const uword dim);
  };



//! @}
