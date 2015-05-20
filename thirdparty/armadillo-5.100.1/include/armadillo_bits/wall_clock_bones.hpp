// Copyright (C) 2008-2013 Conrad Sanderson
// Copyright (C) 2008-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup wall_clock
//! @{


//! Class for measuring time intervals
class wall_clock
  {
  public:
  
  inline  wall_clock();
  inline ~wall_clock();
  
  inline void   tic();  //!< start the timer
  inline double toc();  //!< return the number of seconds since the last call to tic()
  
  
  private:
  
  bool valid;
  
  #if defined(ARMA_USE_CXX11) && !defined(ARMA_DONT_USE_CXX11_CHRONO)
    std::chrono::steady_clock::time_point chrono_time1;
  #elif defined(ARMA_HAVE_GETTIMEOFDAY)
    struct timeval posix_time1;
    struct timeval posix_time2;
  #else
    clock_t time1;
  #endif
  };


//! @}
