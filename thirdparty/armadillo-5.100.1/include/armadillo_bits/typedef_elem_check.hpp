// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup typedef_elem
//! @{


namespace junk
  {
  struct arma_elem_size_test
    {
    
    // arma_static_check( (sizeof(size_t) < sizeof(uword)),  ERROR___TYPE_SIZE_T_IS_SMALLER_THAN_UWORD );
    
    arma_static_check( (sizeof(u8) != 1), ERROR___TYPE_U8_HAS_UNSUPPORTED_SIZE );
    arma_static_check( (sizeof(s8) != 1), ERROR___TYPE_S8_HAS_UNSUPPORTED_SIZE );
    
    arma_static_check( (sizeof(u16) != 2), ERROR___TYPE_U16_HAS_UNSUPPORTED_SIZE );
    arma_static_check( (sizeof(s16) != 2), ERROR___TYPE_S16_HAS_UNSUPPORTED_SIZE );
    
    arma_static_check( (sizeof(u32) != 4), ERROR___TYPE_U32_HAS_UNSUPPORTED_SIZE );
    arma_static_check( (sizeof(s32) != 4), ERROR___TYPE_S32_HAS_UNSUPPORTED_SIZE );
    
    #if defined(ARMA_USE_U64S64)
      arma_static_check( (sizeof(u64) != 8), ERROR___TYPE_U64_HAS_UNSUPPORTED_SIZE );
      arma_static_check( (sizeof(s64) != 8), ERROR___TYPE_S64_HAS_UNSUPPORTED_SIZE );
    #endif
    
    arma_static_check( (sizeof(float)  != 4), ERROR___TYPE_FLOAT_HAS_UNSUPPORTED_SIZE );
    arma_static_check( (sizeof(double) != 8), ERROR___TYPE_DOUBLE_HAS_UNSUPPORTED_SIZE );
    
    arma_static_check( (sizeof(std::complex<float>)  != 8),  ERROR___TYPE_COMPLEX_FLOAT_HAS_UNSUPPORTED_SIZE );
    arma_static_check( (sizeof(std::complex<double>) != 16), ERROR___TYPE_COMPLEX_DOUBLE_HAS_UNSUPPORTED_SIZE );
    
    };
  }


//! @}
