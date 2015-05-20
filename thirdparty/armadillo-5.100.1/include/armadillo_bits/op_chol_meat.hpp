// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_chol
//! @{



template<typename T1>
inline
void
op_chol::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_chol>& X)
  {
  arma_extra_debug_sigprint();
  
  const bool status = auxlib::chol(out, X.m, X.aux_uword_a);
  
  if(status == false)
    {
    out.reset();
    arma_bad("chol(): failed to converge");
    }
  }



//! @}
