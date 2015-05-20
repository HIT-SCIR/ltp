// Copyright (C) 2012-2014 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup spop_strans
//! @{


//! simple transpose operation (no complex conjugates) for sparse matrices 

class spop_strans
  {
  public:
  
  template<typename eT>
  arma_hot inline static void apply_spmat(SpMat<eT>& out, const SpMat<eT>& X);
  
  template<typename T1>
  arma_hot inline static void apply_proxy(SpMat<typename T1::elem_type>& out, const T1& X);
  
  template<typename T1>
  arma_hot inline static void apply(SpMat<typename T1::elem_type>& out, const SpOp<T1,spop_strans>& in);
  
  template<typename T1>
  arma_hot inline static void apply(SpMat<typename T1::elem_type>& out, const SpOp<T1,spop_htrans>& in);
  };



//! @}
