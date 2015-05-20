// Copyright (C) 2008-2010 Conrad Sanderson
// Copyright (C) 2008-2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup operator_cube_schur
//! @{


// operator %, which we define it to do a schur product (element-wise multiplication)


//! element-wise multiplication of BaseCube objects with same element type
template<typename T1, typename T2>
arma_inline
const eGlueCube<T1, T2, eglue_schur>
operator%
  (
  const BaseCube<typename T1::elem_type,T1>& X,
  const BaseCube<typename T1::elem_type,T2>& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return eGlueCube<T1, T2, eglue_schur>(X.get_ref(), Y.get_ref());
  }



//! element-wise multiplication of BaseCube objects with different element types
template<typename T1, typename T2>
inline
const mtGlueCube<typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, glue_mixed_schur>
operator%
  (
  const BaseCube< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T1_result, T1>& X,
  const BaseCube< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T2_result, T2>& Y
  )
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  return mtGlueCube<out_eT, T1, T2, glue_mixed_schur>( X.get_ref(), Y.get_ref() );
  }



//! @}
