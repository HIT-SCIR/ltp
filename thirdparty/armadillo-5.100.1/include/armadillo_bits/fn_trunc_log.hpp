// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_trunc_log
//! @{



template<typename eT>
inline
static
typename arma_real_only<eT>::result
trunc_log(const eT x)
  {
  if(std::numeric_limits<eT>::is_iec559)
    {
    if(x == std::numeric_limits<eT>::infinity())
      {
      return Math<eT>::log_max();
      }
    else
      {
      return (x <= eT(0)) ? Math<eT>::log_min() : std::log(x);
      }
    }
  else
    {
    return std::log(x);
    }
  }



template<typename eT>
inline
static
typename arma_integral_only<eT>::result
trunc_log(const eT x)
  {
  return eT( trunc_log( double(x) ) );
  }



template<typename T>
inline
static
std::complex<T>
trunc_log(const std::complex<T>& x)
  {
  return std::complex<T>( trunc_log( std::abs(x) ), std::arg(x) );
  }



template<typename T1>
arma_inline
const eOp<T1, eop_trunc_log>
trunc_log(const Base<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOp<T1, eop_trunc_log>(A.get_ref());
  }



template<typename T1>
arma_inline
const eOpCube<T1, eop_trunc_log>
trunc_log(const BaseCube<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return eOpCube<T1, eop_trunc_log>(A.get_ref());
  }



//! @}
