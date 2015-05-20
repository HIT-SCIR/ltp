// Copyright (C) 2009-2011 Conrad Sanderson
// Copyright (C) 2009-2011 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_stddev
//! @{

//! Class for finding the standard deviation
class op_stddev
  {
  public:
  
  template<typename T1>
  inline static void apply(Mat<typename T1::pod_type>& out, const mtOp<typename T1::pod_type, T1, op_stddev>& in);
  };

//! @}
