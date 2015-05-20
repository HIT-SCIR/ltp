// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup spop_htrans
//! @{


//! hermitian transpose operation for sparse matrices

class spop_htrans
  {
  public:
  
  template<typename T1>
  arma_hot inline static void apply(SpMat<typename T1::elem_type>& out, const SpOp<T1,spop_htrans>& in, const typename arma_not_cx<typename T1::elem_type>::result* junk = 0);
  
  template<typename T1>
  arma_hot inline static void apply(SpMat<typename T1::elem_type>& out, const SpOp<T1,spop_htrans>& in, const typename arma_cx_only<typename T1::elem_type>::result* junk = 0);
  };



//! @}
