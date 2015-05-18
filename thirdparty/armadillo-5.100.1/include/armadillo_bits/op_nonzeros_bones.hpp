// Copyright (C) 2015 Conrad Sanderson
// Copyright (C) 2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup op_nonzeros
//! @{



class op_nonzeros
  {
  public:
  
  // for dense matrices
  
  template<typename T1>
  static inline void apply_noalias(Mat<typename T1::elem_type>& out, const Proxy<T1>& P);
  
  template<typename T1>
  static inline void apply(Mat<typename T1::elem_type>& out, const Op<T1, op_nonzeros>& X);
  
  
  // for sparse matrices
  
  template<typename T1>
  static inline void apply_noalias(Mat<typename T1::elem_type>& out, const SpBase<typename T1::elem_type, T1>& X);
  };



//! @}
