// Copyright (C) 2011 Conrad Sanderson
// Copyright (C) 2011 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup op_resize
//! @{



class op_resize
  {
  public:
  
  template<typename T1> inline static void apply( Mat<typename T1::elem_type>& out, const     Op<T1,op_resize>& in);
  template<typename T1> inline static void apply(Cube<typename T1::elem_type>& out, const OpCube<T1,op_resize>& in);
  };



//! @}
