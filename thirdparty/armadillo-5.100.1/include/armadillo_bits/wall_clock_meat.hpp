// Copyright (C) 2008-2013 Conrad Sanderson
// Copyright (C) 2008-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup wall_clock
//! @{


inline
wall_clock::wall_clock()
  : valid(false)
  {
  arma_extra_debug_sigprint();
  }



inline
wall_clock::~wall_clock()
  {
  arma_extra_debug_sigprint();
  }



inline
void
wall_clock::tic()
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_CXX11) && !defined(ARMA_DONT_USE_CXX11_CHRONO)
    {
    chrono_time1 = std::chrono::steady_clock::now();
    valid = true;
    }
  #elif defined(ARMA_HAVE_GETTIMEOFDAY)
    {
    gettimeofday(&posix_time1, 0);
    valid = true;
    }
  #else
    {
    time1 = clock();
    valid = true;
    }
  #endif
  }



inline
double
wall_clock::toc()
  {
  arma_extra_debug_sigprint();
  
  if(valid)
    {
    #if defined(ARMA_USE_CXX11) && !defined(ARMA_DONT_USE_CXX11_CHRONO)
      {
      const std::chrono::steady_clock::time_point chrono_time2 = std::chrono::steady_clock::now();
      
      typedef std::chrono::duration<double> duration_type;
      
      const duration_type chrono_span = std::chrono::duration_cast< duration_type >(chrono_time2 - chrono_time1);
      
      return chrono_span.count();
      }
    #elif defined(ARMA_HAVE_GETTIMEOFDAY)
      {
      gettimeofday(&posix_time2, 0);
      
      const double tmp_time1 = posix_time1.tv_sec + posix_time1.tv_usec * 1.0e-6;
      const double tmp_time2 = posix_time2.tv_sec + posix_time2.tv_usec * 1.0e-6;
      
      return tmp_time2 - tmp_time1;
      }
    #else
      {
      clock_t time2 = clock();
      
      clock_t diff = time2 - time1;
      
      return double(diff) / double(CLOCKS_PER_SEC);
      }
    #endif
    }
  else
    {  
    return 0.0;
    }
  }



//! @}

