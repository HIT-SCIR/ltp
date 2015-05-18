// Copyright (C) 2012 Conrad Sanderson
// Copyright (C) 2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup op_hist
//! @{



class op_hist
  {
  public:
  
  template<typename T1>
  inline static void apply(Mat<uword>& out, const mtOp<uword, T1, op_hist>& X);
  };



//! @}
