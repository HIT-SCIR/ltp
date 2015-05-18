// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup SpGlue
//! @{



template<typename T1, typename T2, typename spglue_type>
inline
SpGlue<T1,T2,spglue_type>::SpGlue(const T1& in_A, const T2& in_B)
  : A(in_A)
  , B(in_B)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename T2, typename spglue_type>
inline
SpGlue<T1,T2,spglue_type>::SpGlue(const T1& in_A, const T2& in_B, const typename T1::elem_type in_aux)
  : A(in_A)
  , B(in_B)
  , aux(in_aux)
  {
  arma_extra_debug_sigprint();
  }



template<typename T1, typename T2, typename spglue_type>
inline
SpGlue<T1,T2,spglue_type>::~SpGlue()
  {
  arma_extra_debug_sigprint();
  }



//! @}
