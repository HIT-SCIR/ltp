// Copyright (C) 2014 Conrad Sanderson
// Copyright (C) 2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


#if defined(ARMA_USE_HDF5)
  #if !defined(ARMA_HDF5_INCLUDE_DIR)
    #include <hdf5.h>
  #else
    #define ARMA_STR1(x) x
    #define ARMA_STR2(x) ARMA_STR1(x)
    
    #define ARMA_HDF5_HEADER ARMA_STR2(ARMA_HDF5_INCLUDE_DIR)ARMA_STR2(hdf5.h)
    
    #include ARMA_INCFILE_WRAP(ARMA_HDF5_HEADER)
    
    #undef ARMA_STR1
    #undef ARMA_STR2
    #undef ARMA_HDF5_HEADER
  #endif

  #if defined(H5_USE_16_API_DEFAULT) || defined(H5_USE_16_API)
    #pragma message ("WARNING: disabling use of HDF5 due to its incompatible configuration")
    #undef ARMA_USE_HDF5
  #endif
#endif
