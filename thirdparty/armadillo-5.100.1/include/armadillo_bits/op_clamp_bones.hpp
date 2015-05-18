// Copyright (C) 2014 Conrad Sanderson
// Copyright (C) 2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup op_clamp
//! @{



class op_clamp
  {
  public:
  
  template<typename T1> inline static void apply(Mat<typename T1::elem_type>& out, const mtOp<typename T1::elem_type, T1, op_clamp>& in);
  
  template<typename T1> inline static void apply_noalias(Mat<typename T1::elem_type>& out, const Proxy<T1>& P, const typename T1::elem_type min_val, const typename T1::elem_type max_val);
  };



//! @}
