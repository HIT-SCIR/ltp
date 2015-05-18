// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup spglue_minus
//! @{



class spglue_minus
  {
  public:
  
  template<typename T1, typename T2>
  arma_hot inline static void apply(SpMat<typename T1::elem_type>& out, const SpGlue<T1,T2,spglue_minus>& X);
  
  template<typename eT, typename T1, typename T2>
  arma_hot inline static void apply_noalias(SpMat<eT>& result, const SpProxy<T1>& pa, const SpProxy<T2>& pb);
  };



class spglue_minus2
  {
  public:
  
  template<typename T1, typename T2>
  arma_hot inline static void apply(SpMat<typename T1::elem_type>& out, const SpGlue<T1,T2,spglue_minus2>& X);
  };



//! @}

