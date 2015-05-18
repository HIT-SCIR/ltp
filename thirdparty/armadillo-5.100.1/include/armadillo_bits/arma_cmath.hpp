// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup arma_cmath
//! @{



//
// wrappers for isfinite


template<typename eT>
arma_inline
bool
arma_isfinite(eT val)
  {
  arma_ignore(val);
    
  return true;
  }



template<>
arma_inline
bool
arma_isfinite(float x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::isfinite(x);
    }
  #elif defined(ARMA_HAVE_TR1)
    {
    return std::tr1::isfinite(x);
    }
  #elif defined(ARMA_HAVE_ISFINITE)
    {
    return (std::isfinite(x) != 0);
    }
  #else
    {
    const float y = (std::numeric_limits<float>::max)();
    
    const volatile float xx = x;
    
    return (xx == xx) && (x >= -y) && (x <= y);
    }
  #endif
  }



template<>
arma_inline
bool
arma_isfinite(double x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::isfinite(x);
    }
  #elif defined(ARMA_HAVE_TR1)
    {
    return std::tr1::isfinite(x);
    }
  #elif defined(ARMA_HAVE_ISFINITE)
    {
    return (std::isfinite(x) != 0);
    }
  #else
    {
    const double y = (std::numeric_limits<double>::max)();
    
    const volatile double xx = x;
    
    return (xx == xx) && (x >= -y) && (x <= y);
    }
  #endif
  }



template<typename T>
arma_inline
bool
arma_isfinite(const std::complex<T>& x)
  {
  if( (arma_isfinite(x.real()) == false) || (arma_isfinite(x.imag()) == false) )
    {
    return false;
    }
  else
    {
    return true;
    }
  }



//
// wrappers for isinf


template<typename eT>
arma_inline
bool
arma_isinf(eT val)
  {
  arma_ignore(val);
    
  return false;
  }



template<>
arma_inline
bool
arma_isinf(float x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::isinf(x);
    }
  #elif defined(ARMA_HAVE_ISINF)
    {
    return (std::isinf(x) != 0);
    }
  #else
    {
    const float y = (std::numeric_limits<float>::max)();
    
    const volatile float xx = x;
    
    return (xx == xx) && ((x < -y) || (x > y));
    }
  #endif
  }



template<>
arma_inline
bool
arma_isinf(double x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::isinf(x);
    }
  #elif defined(ARMA_HAVE_ISINF)
    {
    return (std::isinf(x) != 0);
    }
  #else
    {
    const double y = (std::numeric_limits<double>::max)();
    
    const volatile double xx = x;
    
    return (xx == xx) && ((x < -y) || (x > y));
    }
  #endif
  }



template<typename T>
arma_inline
bool
arma_isinf(const std::complex<T>& x)
  {
  return ( arma_isinf(x.real()) || arma_isinf(x.imag()) );
  }



//
// wrappers for isnan


template<typename eT>
arma_inline
bool
arma_isnan(eT val)
  {
  arma_ignore(val);
    
  return false;
  }



template<>
arma_inline
bool
arma_isnan(float x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::isnan(x);
    }
  #elif defined(ARMA_HAVE_ISNAN)
    {
    return (std::isnan(x) != 0);
    }
  #else
    {
    const volatile float xx = x;
    
    return (xx != xx);
    }
  #endif
  }



template<>
arma_inline
bool
arma_isnan(double x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::isnan(x);
    }
  #elif defined(ARMA_HAVE_ISNAN)
    {
    return (std::isnan(x) != 0);
    }
  #else
    {
    const volatile double xx = x;
    
    return (xx != xx);
    }
  #endif
  }



template<typename T>
arma_inline
bool
arma_isnan(const std::complex<T>& x)
  {
  return ( arma_isnan(x.real()) || arma_isnan(x.imag()) );
  }



// rudimentary wrappers for log1p()

arma_inline
float
arma_log1p(const float x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::log1p(x);
    }
  #else
    {
    if((x >= float(0)) && (x < std::numeric_limits<float>::epsilon()))
      {
      return x;
      }
    else
    if((x < float(0)) && (-x < std::numeric_limits<float>::epsilon()))
      {
      return x;
      }
    else
      {
      return std::log(float(1) + x);
      }
    }
  #endif
  }



arma_inline
double
arma_log1p(const double x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::log1p(x);
    }
  #elif defined(ARMA_HAVE_LOG1P)
    {
    return log1p(x);
    }
  #else
    {
    if((x >= double(0)) && (x < std::numeric_limits<double>::epsilon()))
      {
      return x;
      }
    else
    if((x < double(0)) && (-x < std::numeric_limits<double>::epsilon()))
      {
      return x;
      }
    else
      {
      return std::log(double(1) + x);
      }
    }
  #endif
  }





//
// wrappers for trigonometric functions
// 
// wherever possible, try to use C++11 or TR1 versions of the following functions:
// 
// complex acos
// complex asin
// complex atan
//
// real    acosh
// real    asinh
// real    atanh
//
// complex acosh
// complex asinh
// complex atanh
// 
// 
// if C++11 or TR1 are not available, we have rudimentary versions of:
// 
// real    acosh
// real    asinh
// real    atanh



template<typename T>
arma_inline
std::complex<T>
arma_acos(const std::complex<T>& x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::acos(x);
    }
  #elif defined(ARMA_HAVE_TR1)
    {
    return std::tr1::acos(x);
    }
  #else
    {
    arma_ignore(x);
    arma_stop("acos(): need C++11 compiler");
    
    return std::complex<T>(0);
    }
  #endif
  }



template<typename T>
arma_inline
std::complex<T>
arma_asin(const std::complex<T>& x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::asin(x);
    }
  #elif defined(ARMA_HAVE_TR1)
    {
    return std::tr1::asin(x);
    }
  #else
    {
    arma_ignore(x);
    arma_stop("asin(): need C++11 compiler");
    
    return std::complex<T>(0);
    }
  #endif
  }



template<typename T>
arma_inline
std::complex<T>
arma_atan(const std::complex<T>& x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::atan(x);
    }
  #elif defined(ARMA_HAVE_TR1)
    {
    return std::tr1::atan(x);
    }
  #else
    {
    arma_ignore(x);
    arma_stop("atan(): need C++11 compiler");
    
    return std::complex<T>(0);
    }
  #endif
  }



template<typename eT>
arma_inline
eT
arma_acosh(const eT x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::acosh(x);
    }
  #elif defined(ARMA_HAVE_TR1)
    {
    return std::tr1::acosh(x);
    }
  #else
    {
    if(x >= eT(1))
      {
      // http://functions.wolfram.com/ElementaryFunctions/ArcCosh/02/
      return std::log( x + std::sqrt(x*x - eT(1)) );
      }
    else
      {
      if(std::numeric_limits<eT>::has_quiet_NaN == true)
        {
        return -(std::numeric_limits<eT>::quiet_NaN());
        }
      else
        {
        return eT(0);
        }
      }
    }
  #endif
  }



template<typename eT>
arma_inline
eT
arma_asinh(const eT x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::asinh(x);
    }
  #elif defined(ARMA_HAVE_TR1)
    {
    return std::tr1::asinh(x);
    }
  #else
    {
    // http://functions.wolfram.com/ElementaryFunctions/ArcSinh/02/
    return std::log( x + std::sqrt(x*x + eT(1)) );
    }
  #endif
  }



template<typename eT>
arma_inline
eT
arma_atanh(const eT x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::atanh(x);
    }
  #elif defined(ARMA_HAVE_TR1)
    {
    return std::tr1::atanh(x);
    }
  #else
    {
    if( (x >= eT(-1)) && (x <= eT(+1)) )
      {
      // http://functions.wolfram.com/ElementaryFunctions/ArcTanh/02/
      return std::log( ( eT(1)+x ) / ( eT(1)-x ) ) / eT(2);
      }
    else
      {
      if(std::numeric_limits<eT>::has_quiet_NaN == true)
        {
        return -(std::numeric_limits<eT>::quiet_NaN());
        }
      else
        {
        return eT(0);
        }
      }
    }
  #endif
  }



template<typename T>
arma_inline
std::complex<T>
arma_acosh(const std::complex<T>& x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::acosh(x);
    }
  #elif defined(ARMA_HAVE_TR1)
    {
    return std::tr1::acosh(x);
    }
  #else
    {
    arma_ignore(x);
    arma_stop("acosh(): need C++11 compiler");
    
    return std::complex<T>(0);
    }
  #endif
  }



template<typename T>
arma_inline
std::complex<T>
arma_asinh(const std::complex<T>& x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::asinh(x);
    }
  #elif defined(ARMA_HAVE_TR1)
    {
    return std::tr1::asinh(x);
    }
  #else
    {
    arma_ignore(x);
    arma_stop("asinh(): need C++11 compiler");
    
    return std::complex<T>(0);
    }
  #endif
  }



template<typename T>
arma_inline
std::complex<T>
arma_atanh(const std::complex<T>& x)
  {
  #if defined(ARMA_USE_CXX11)
    {
    return std::atanh(x);
    }
  #elif defined(ARMA_HAVE_TR1)
    {
    return std::tr1::atanh(x);
    }
  #else
    {
    arma_ignore(x);
    arma_stop("atanh(): need C++11 compiler");
    
    return std::complex<T>(0);
    }
  #endif
  }



//! @}
