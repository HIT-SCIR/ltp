// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_diagvec
//! @{



class op_diagvec
  {
  public:
  
  template<typename T1>
  inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_diagvec>& X);
  
  template<typename T1>
  arma_hot inline static void apply_unwrap(Mat<typename T1::elem_type>& out, const T1& X,       const uword row_offset, const uword col_offset, const uword len);
  
  template<typename T1>
  arma_hot inline static void apply_proxy(Mat<typename T1::elem_type>& out, const Proxy<T1>& P, const uword row_offset, const uword col_offset, const uword len);
  };



//! @}
