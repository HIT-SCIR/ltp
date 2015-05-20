// Copyright (C) 2010-2015 Conrad Sanderson
// Copyright (C) 2010-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_cumsum
//! @{



class op_cumsum_mat
  {
  public:
  
  template<typename eT>
  inline static void apply_noalias(Mat<eT>& out, const Mat<eT>& X, const uword dim);
  
  template<typename T1>
  inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_cumsum_mat>& in);
  };



class op_cumsum_vec
  {
  public:
  
  template<typename eT>
  inline static void apply_noalias(Mat<eT>& out, const Mat<eT>& X);
  
  template<typename T1>
  inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_cumsum_vec>& in);
  };

//! @}
