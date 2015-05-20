// Copyright (C) 2012 NICTA (www.nicta.com.au)
// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup cond_rel
//! @{



template<>
template<typename eT>
arma_inline
bool
cond_rel<true>::lt(const eT A, const eT B)
  {
  return (A < B);
  }
  


template<>
template<typename eT>
arma_inline
bool
cond_rel<false>::lt(const eT, const eT)
  {
  return false;
  }
  


template<>
template<typename eT>
arma_inline
bool
cond_rel<true>::gt(const eT A, const eT B)
  {
  return (A > B);
  }
  


template<>
template<typename eT>
arma_inline
bool
cond_rel<false>::gt(const eT, const eT)
  {
  return false;
  }
  


template<>
template<typename eT>
arma_inline
bool
cond_rel<true>::leq(const eT A, const eT B)
  {
  return (A <= B);
  }
  


template<>
template<typename eT>
arma_inline
bool
cond_rel<false>::leq(const eT, const eT)
  {
  return false;
  }
  


template<>
template<typename eT>
arma_inline
bool
cond_rel<true>::geq(const eT A, const eT B)
  {
  return (A >= B);
  }
  


template<>
template<typename eT>
arma_inline
bool
cond_rel<false>::geq(const eT, const eT)
  {
  return false;
  }
  


//! @}
