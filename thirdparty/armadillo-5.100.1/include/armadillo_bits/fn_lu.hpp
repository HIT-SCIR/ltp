// Copyright (C) 2008-2011 Conrad Sanderson
// Copyright (C) 2008-2011 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_lu
//! @{



//! immediate lower upper decomposition, permutation info is embedded into L (similar to Matlab/Octave)
template<typename T1>
inline
bool
lu
  (
         Mat<typename T1::elem_type>&    L,
         Mat<typename T1::elem_type>&    U,
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  arma_debug_check( (&L == &U), "lu(): L and U are the same object");
  
  const bool status = auxlib::lu(L, U, X);
  
  if(status == false)
    {
    L.reset();
    U.reset();
    arma_bad("lu(): failed to converge", false);
    }
  
  return status;
  }



//! immediate lower upper decomposition, also providing the permutation matrix
template<typename T1>
inline
bool
lu
  (
         Mat<typename T1::elem_type>&    L,
         Mat<typename T1::elem_type>&    U, 
         Mat<typename T1::elem_type>&    P,
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  arma_debug_check( ( (&L == &U) || (&L == &P) || (&U == &P) ), "lu(): two or more output objects are the same object");
  
  const bool status = auxlib::lu(L, U, P, X);
  
  if(status == false)
    {
    L.reset();
    U.reset();
    P.reset();
    arma_bad("lu(): failed to converge", false);
    }
  
  return status;
  }



//! @}
