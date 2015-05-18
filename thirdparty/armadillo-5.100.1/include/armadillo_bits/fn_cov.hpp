// Copyright (C) 2009-2010 Conrad Sanderson
// Copyright (C) 2009-2010 NICTA (www.nicta.com.au)
// Copyright (C) 2009-2010 Dimitrios Bouzas
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_cov
//! @{



template<typename T1>
inline
const Op<T1, op_cov>
cov(const Base<typename T1::elem_type,T1>& X, const uword norm_type = 0)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (norm_type > 1), "cov(): norm_type must be 0 or 1");

  return Op<T1, op_cov>(X.get_ref(), norm_type, 0);
  }



template<typename T1, typename T2>
inline
const Glue<T1,T2,glue_cov>
cov(const Base<typename T1::elem_type,T1>& A, const Base<typename T1::elem_type,T2>& B, const uword norm_type = 0)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (norm_type > 1), "cov(): norm_type must be 0 or 1");
  
  return Glue<T1, T2, glue_cov>(A.get_ref(), B.get_ref(), norm_type);
  }



//! @}
