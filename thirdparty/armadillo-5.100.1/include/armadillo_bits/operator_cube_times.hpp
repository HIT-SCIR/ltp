// Copyright (C) 2008-2010 Conrad Sanderson
// Copyright (C) 2008-2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.




//! \addtogroup operator_cube_times
//! @{



//! BaseCube * scalar
template<typename T1>
arma_inline
const eOpCube<T1, eop_scalar_times>
operator*
  (
  const BaseCube<typename T1::elem_type,T1>& X,
  const typename T1::elem_type               k
  )
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_scalar_times>(X.get_ref(), k);
  }



//! scalar * BaseCube
template<typename T1>
arma_inline
const eOpCube<T1, eop_scalar_times>
operator*
  (
  const typename T1::elem_type               k,
  const BaseCube<typename T1::elem_type,T1>& X
  )
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_scalar_times>(X.get_ref(), k);
  }



//! non-complex BaseCube * complex scalar (experimental)
template<typename T1>
arma_inline
const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_times>
operator*
  (
  const BaseCube<typename T1::pod_type, T1>& X,
  const std::complex<typename T1::pod_type>& k
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_times>('j', X.get_ref(), k);
  }



//! complex scalar * non-complex BaseCube (experimental)
template<typename T1>
arma_inline
const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_times>
operator*
  (
  const std::complex<typename T1::pod_type>& k,
  const BaseCube<typename T1::pod_type, T1>& X
  )
  {
  arma_extra_debug_sigprint();
  
  return mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_times>('j', X.get_ref(), k);
  }



//! @}
