// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_toeplitz
//! @{



template<typename T1>
inline
Op<T1, op_toeplitz>
toeplitz(const Base<typename T1::elem_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_toeplitz>( X.get_ref() );
  }



template<typename T1>
inline
Op<T1, op_toeplitz_c>
circ_toeplitz(const Base<typename T1::elem_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_toeplitz_c>( X.get_ref() );
  }



template<typename T1, typename T2>
inline
Glue<T1, T2, glue_toeplitz>
toeplitz(const Base<typename T1::elem_type,T1>& X, const Base<typename T1::elem_type,T2>& Y)
  {
  arma_extra_debug_sigprint();
  
  return Glue<T1, T2, glue_toeplitz>( X.get_ref(), Y.get_ref() );
  }



//! @}
