// Copyright (C) 2012 Ryan Curtin
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup mtSpOp
//! @{

// Class for delayed multi-type sparse operations.  These are operations where
// the resulting type is different than the stored type.



template<typename out_eT, typename T1, typename op_type>
class mtSpOp : public SpBase<out_eT, mtSpOp<out_eT, T1, op_type> >
  {
  public:

  typedef          out_eT                       elem_type;
  typedef typename get_pod_type<out_eT>::result pod_type;

  typedef typename T1::elem_type                in_eT;

  static const bool is_row = false;
  static const bool is_col = false;

  inline explicit mtSpOp(const T1& in_m);
  inline          mtSpOp(const T1& in_m, const uword aux_uword_a, const uword aux_uword_b);

  inline          ~mtSpOp();

  arma_aligned const T1&    m;
  arma_aligned       uword  aux_uword_a;
  arma_aligned       uword  aux_uword_b;
  };



//! @}
