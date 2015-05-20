// Copyright (C) 2014 Conrad Sanderson
// Copyright (C) 2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_clamp
//! @{



template<typename T1>
inline
typename
enable_if2
  <
  is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no,
  const mtOp<typename T1::elem_type, T1, op_clamp>
  >::result
clamp(const T1& X, const typename T1::elem_type min_val, const typename T1::elem_type max_val)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (min_val > max_val), "clamp(): min_val has to be smaller than max_val" );
  
  return mtOp<typename T1::elem_type, T1, op_clamp>(mtOp_dual_aux_indicator(), X, min_val, max_val);
  }



//! @}
