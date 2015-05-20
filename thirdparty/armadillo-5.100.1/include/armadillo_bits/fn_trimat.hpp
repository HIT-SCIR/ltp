// Copyright (C) 2010 Conrad Sanderson
// Copyright (C) 2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_trimat
//! @{


template<typename T1>
arma_inline
const Op<T1, op_trimat>
trimatu(const Base<typename T1::elem_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_trimat>(X.get_ref(), 0, 0);
  }



template<typename T1>
arma_inline
const Op<T1, op_trimat>
trimatl(const Base<typename T1::elem_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_trimat>(X.get_ref(), 1, 0);
  }



//! @}
