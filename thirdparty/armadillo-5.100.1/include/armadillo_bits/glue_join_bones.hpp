// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup glue_join
//! @{



class glue_join
  {
  public:
  
  template<typename T1, typename T2>
  inline static void apply(Mat<typename T1::elem_type>& out, const Glue<T1,T2,glue_join>& X);
  
  template<typename eT>
  inline static void apply_noalias(Mat<eT>& out, const Mat<eT>& A, const Mat<eT>& B, const uword join_type);
  
  template<typename T1, typename T2>
  inline static void apply(Cube<typename T1::elem_type>& out, const GlueCube<T1,T2,glue_join>& X);
  };



//! @}

