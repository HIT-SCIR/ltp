// Copyright (C) 2009-2012 Conrad Sanderson
// Copyright (C) 2009-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_prod
//! @{

//! Class for finding products of values in a matrix  (e.g. along rows or columns)
class op_prod
  {
  public:
  
  template<typename T1>
  inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1, op_prod>& in);
  
  template<typename eT>
  inline static eT prod(const subview<eT>& S);
  
  template<typename T1>
  inline static typename T1::elem_type prod(const Base<typename T1::elem_type,T1>& X);
  };


//! @}
