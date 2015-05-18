// Copyright (C) 2008-2011 Conrad Sanderson
// Copyright (C) 2008-2011 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup mtOpCube
//! @{



template<typename out_eT, typename T1, typename op_type>
class mtOpCube : public BaseCube<out_eT, mtOpCube<out_eT, T1, op_type> >
  {
  public:
  
  typedef          out_eT                       elem_type;
  typedef typename get_pod_type<out_eT>::result pod_type;

  typedef typename T1::elem_type                in_eT;

  inline explicit mtOpCube(const T1& in_m);
  inline          mtOpCube(const T1& in_m, const in_eT in_aux);
  inline          mtOpCube(const T1& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c);
  inline          mtOpCube(const T1& in_m, const in_eT in_aux,         const uword in_aux_uword_a, const uword in_aux_uword_b, const uword in_aux_uword_c);
  
  inline          mtOpCube(const char junk, const T1& in_m, const out_eT in_aux);
  
  inline         ~mtOpCube();
    
  
  arma_aligned const T1&    m;            //!< storage of reference to the operand (eg. a matrix)
  arma_aligned       in_eT  aux;          //!< storage of auxiliary data, using the element type as used by T1
  arma_aligned       out_eT aux_out_eT;   //!< storage of auxiliary data, using the element type as specified by the out_eT template parameter
  arma_aligned       uword  aux_uword_a;  //!< storage of auxiliary data, uword format
  arma_aligned       uword  aux_uword_b;  //!< storage of auxiliary data, uword format
  arma_aligned       uword  aux_uword_c;  //!< storage of auxiliary data, uword format
  
  };



//! @}
