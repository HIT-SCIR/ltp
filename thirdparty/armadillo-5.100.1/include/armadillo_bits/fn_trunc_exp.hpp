// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup fn_trunc_exp
//! @{



template<typename eT>
inline
static
typename arma_real_only<eT>::result
trunc_exp(const eT x)
  {
  if(std::numeric_limits<eT>::is_iec559 && (x >= Math<eT>::log_max() ))
    {
    return std::numeric_limits<eT>::max();
    }
  else
    {
    return std::exp(x);
    }
  }



template<typename eT>
inline
static
typename arma_integral_only<eT>::result
trunc_exp(const eT x)
  {
  return eT( trunc_exp( double(x) ) );
  }



template<typename T>
inline
static
std::complex<T>
trunc_exp(const std::complex<T>& x)
  {
  return std::polar( trunc_exp( x.real() ), x.imag() );
  }



template<typename T1>
arma_inline
const eOp<T1, eop_trunc_exp>
trunc_exp(const Base<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_trunc_exp>(A.get_ref());
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_trunc_exp>
trunc_exp(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_trunc_exp>(A.get_ref());
  }



//! @}
