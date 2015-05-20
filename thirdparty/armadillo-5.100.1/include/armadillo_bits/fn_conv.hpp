// Copyright (C) 2010 Conrad Sanderson
// Copyright (C) 2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_conv
//! @{



//! Convolution, which is also equivalent to polynomial multiplication and FIR digital filtering.

template<typename T1, typename T2>
inline
const Glue<T1, T2, glue_conv>
conv(const Base<typename T1::elem_type,T1>& A, const Base<typename T1::elem_type,T2>& B)
  {
  arma_extra_debug_sigprint();
  
  return Glue<T1, T2, glue_conv>(A.get_ref(), B.get_ref());
  }



//! @}
