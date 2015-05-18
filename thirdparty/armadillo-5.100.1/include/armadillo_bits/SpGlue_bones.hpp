// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup SpGlue
//! @{



template<typename T1, typename T2, typename spglue_type>
class SpGlue : public SpBase<typename T1::elem_type, SpGlue<T1, T2, spglue_type> >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  static const bool is_row = ( (T1::is_row || T2::is_row) && is_spglue_elem<spglue_type>::value ) || ( (is_spglue_times<spglue_type>::value || is_spglue_times2<spglue_type>::value) ? T1::is_row : false );
  static const bool is_col = ( (T1::is_col || T2::is_col) && is_spglue_elem<spglue_type>::value ) || ( (is_spglue_times<spglue_type>::value || is_spglue_times2<spglue_type>::value) ? T2::is_col : false );
  
  arma_inline  SpGlue(const T1& in_A, const T2& in_B);
  arma_inline  SpGlue(const T1& in_A, const T2& in_B, const elem_type in_aux);
  arma_inline ~SpGlue();
  
  const T1&       A;    //!< first operand
  const T2&       B;    //!< second operand
        elem_type aux;
  };



//! @}
