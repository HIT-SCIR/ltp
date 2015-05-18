// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup op_reshape
//! @{



class op_reshape
  {
  public:
  
  template<typename eT> inline static void apply_unwrap(Mat<eT>&                     out, const Mat<eT>&   A, const uword in_n_rows, const uword in_n_cols, const uword in_dim);
  
  template<typename T1> inline static void apply_proxy (Mat<typename T1::elem_type>& out, const Proxy<T1>& P, const uword in_n_rows, const uword in_n_cols);
  
  template<typename T1> inline static void apply       (Mat<typename T1::elem_type>& out, const Op<T1,op_reshape>& in);
  };



class op_reshape_ext
  {
  public:
  
  template<typename T1> inline static void apply( Mat<typename T1::elem_type>& out, const     Op<T1,op_reshape_ext>& in);
  template<typename T1> inline static void apply(Cube<typename T1::elem_type>& out, const OpCube<T1,op_reshape_ext>& in);
  };



//! @}
