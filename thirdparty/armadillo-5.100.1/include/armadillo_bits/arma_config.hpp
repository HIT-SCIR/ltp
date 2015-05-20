// Copyright (C) 2008-2015 Conrad Sanderson
// Copyright (C) 2008-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup arma_config
//! @{



struct arma_config
  {
  #if defined(ARMA_MAT_PREALLOC)
    static const uword mat_prealloc = (sword(ARMA_MAT_PREALLOC) > 0) ? uword(ARMA_MAT_PREALLOC) : 1;
  #else
    static const uword mat_prealloc = 16;
  #endif
  
  
  #if defined(ARMA_SPMAT_CHUNKSIZE)
    static const uword spmat_chunksize = (sword(ARMA_SPMAT_CHUNKSIZE) > 0) ? uword(ARMA_SPMAT_CHUNKSIZE) : 256;
  #else
    static const uword spmat_chunksize = 256;
  #endif
  
  
  #if defined(ARMA_USE_ATLAS)
    static const bool atlas = true;
  #else
    static const bool atlas = false;
  #endif
  
  
  #if defined(ARMA_USE_LAPACK)
    static const bool lapack = true;
  #else
    static const bool lapack = false;
  #endif
  
  
  #if defined(ARMA_USE_BLAS)
    static const bool blas = true;
  #else
    static const bool blas = false;
  #endif
  
  
  #if defined(ARMA_USE_ARPACK)
    static const bool arpack = true;
  #else
    static const bool arpack = false;
  #endif
  
  
  #if defined(ARMA_USE_SUPERLU)
    static const bool superlu = true;
  #else
    static const bool superlu = false;
  #endif
  
  
  #if defined(ARMA_USE_HDF5)
    static const bool hdf5 = true;
  #else
    static const bool hdf5 = false;
  #endif
  
  
  #if defined(ARMA_NO_DEBUG)
    static const bool debug = false;
  #else
    static const bool debug = true;
  #endif
  
  
  #if defined(ARMA_EXTRA_DEBUG)
    static const bool extra_debug = true;
  #else
    static const bool extra_debug = false;
  #endif
  
  
  #if defined(ARMA_GOOD_COMPILER)
    static const bool good_comp = true;
  #else
    static const bool good_comp = false;
  #endif
  
  
  #if (  \
         defined(ARMA_EXTRA_MAT_PROTO)   || defined(ARMA_EXTRA_MAT_MEAT)   \
      || defined(ARMA_EXTRA_COL_PROTO)   || defined(ARMA_EXTRA_COL_MEAT)   \
      || defined(ARMA_EXTRA_ROW_PROTO)   || defined(ARMA_EXTRA_ROW_MEAT)   \
      || defined(ARMA_EXTRA_CUBE_PROTO)  || defined(ARMA_EXTRA_CUBE_MEAT)  \
      || defined(ARMA_EXTRA_FIELD_PROTO) || defined(ARMA_EXTRA_FIELD_MEAT) \
      || defined(ARMA_EXTRA_SPMAT_PROTO) || defined(ARMA_EXTRA_SPMAT_MEAT) \
      || defined(ARMA_EXTRA_SPCOL_PROTO) || defined(ARMA_EXTRA_SPCOL_MEAT) \
      || defined(ARMA_EXTRA_SPROW_PROTO) || defined(ARMA_EXTRA_SPROW_MEAT) \
      )
    static const bool extra_code = true;
  #else
    static const bool extra_code = false;
  #endif
  
  
  #if defined(ARMA_USE_CXX11)
    static const bool use_cxx11 = true;
  #else
    static const bool use_cxx11 = false;
  #endif
  
  
  #if defined(ARMA_USE_WRAPPER)
    static const bool use_wrapper = true;
  #else
    static const bool use_wrapper = false;
  #endif
  
  
  #if defined(_OPENMP)
    static const bool openmp = true;
  #else
    static const bool openmp = false;
  #endif
  };



//! @}
