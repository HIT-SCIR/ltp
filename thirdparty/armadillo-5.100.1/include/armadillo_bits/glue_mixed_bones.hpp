// Copyright (C) 2009-2010 Conrad Sanderson
// Copyright (C) 2009-2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup glue_mixed
//! @{



class glue_mixed_times
  {
  public:
  
  template<typename T1, typename T2>
  inline static void apply(Mat<typename eT_promoter<T1,T2>::eT>& out, const mtGlue<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_times>& X);
  };



class glue_mixed_plus
  {
  public:
  
  template<typename T1, typename T2>
  inline static void apply(Mat<typename eT_promoter<T1,T2>::eT>& out, const mtGlue<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_plus>& X);
  
  template<typename T1, typename T2>
  inline static void apply(Cube<typename eT_promoter<T1,T2>::eT>& out, const mtGlueCube<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_plus>& X);
  };



class glue_mixed_minus
  {
  public:
  
  template<typename T1, typename T2>
  inline static void apply(Mat<typename eT_promoter<T1,T2>::eT>& out, const mtGlue<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_minus>& X);
  
  template<typename T1, typename T2>
  inline static void apply(Cube<typename eT_promoter<T1,T2>::eT>& out, const mtGlueCube<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_minus>& X);
  };



class glue_mixed_div
  {
  public:
  
  template<typename T1, typename T2>
  inline static void apply(Mat<typename eT_promoter<T1,T2>::eT>& out, const mtGlue<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_div>& X);
  
  template<typename T1, typename T2>
  inline static void apply(Cube<typename eT_promoter<T1,T2>::eT>& out, const mtGlueCube<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_div>& X);
  };



class glue_mixed_schur
  {
  public:
  
  template<typename T1, typename T2>
  inline static void apply(Mat<typename eT_promoter<T1,T2>::eT>& out, const mtGlue<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_schur>& X);
  
  template<typename T1, typename T2>
  inline static void apply(Cube<typename eT_promoter<T1,T2>::eT>& out, const mtGlueCube<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_schur>& X);
  };



//! @}
