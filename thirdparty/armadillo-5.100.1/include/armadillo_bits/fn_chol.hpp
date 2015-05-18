// Copyright (C) 2009-2014 Conrad Sanderson
// Copyright (C) 2009-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_chol
//! @{



template<typename T1>
inline
const Op<T1, op_chol>
chol
  (
  const Base<typename T1::elem_type,T1>& X,
  const char* layout = "upper",
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const char sig = (layout != NULL) ? layout[0] : char(0);
  
  arma_debug_check( ((sig != 'u') && (sig != 'l')), "chol(): layout must be \"upper\" or \"lower\"" );
  
  return Op<T1, op_chol>(X.get_ref(), ((sig == 'u') ? 0 : 1), 0 );
  }



template<typename T1>
inline
bool
chol
  (
         Mat<typename T1::elem_type>&    out,
  const Base<typename T1::elem_type,T1>& X,
  const char* layout = "upper",
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  try
    {
    out = chol(X, layout);
    }
  catch(std::runtime_error&)
    {
    return false;
    }
  
  return true;
  }



//! @}
