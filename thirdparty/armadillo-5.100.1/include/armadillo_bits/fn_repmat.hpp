// Copyright (C) 2009-2015 Conrad Sanderson
// Copyright (C) 2009-2015 NICTA (www.nicta.com.au)
// Copyright (C) 2009-2010 Dimitrios Bouzas
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup fn_repmat
//! @{


template<typename T1>
arma_inline
const Op<T1, op_repmat>
repmat(const Base<typename T1::elem_type,T1>& A, const uword r, const uword c)
  {
  arma_extra_debug_sigprint();

  return Op<T1, op_repmat>(A.get_ref(), r, c);
  }



template<typename T1>
arma_inline
const SpOp<T1, spop_repmat>
repmat(const SpBase<typename T1::elem_type,T1>& A, const uword r, const uword c)
  {
  arma_extra_debug_sigprint();

  return SpOp<T1, spop_repmat>(A.get_ref(), r, c);
  }



//! @}
