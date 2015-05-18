// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup strip
//! @{



template<typename T1>
struct strip_diagmat
  {
  typedef T1 stored_type;
  
  arma_hot inline
  strip_diagmat(const T1& X)
    : M(X)
    {
    arma_extra_debug_sigprint();
    }
  
  static const bool do_diagmat = false;
  
  const T1& M;
  };



template<typename T1>
struct strip_diagmat< Op<T1, op_diagmat> >
  {
  typedef T1 stored_type;
  
  arma_hot inline
  strip_diagmat(const Op<T1, op_diagmat>& X)
    : M(X.m)
    {
    arma_extra_debug_sigprint();
    }
  
  static const bool do_diagmat = true;
  
  const T1& M;
  };



template<typename T1>
struct strip_inv
  {
  typedef T1 stored_type;
  
  arma_hot inline
  strip_inv(const T1& X)
    : M(X)
    {
    arma_extra_debug_sigprint();
    }
  
  const T1& M;
  
  static const bool slow   = false;
  static const bool do_inv = false;
  };



template<typename T1>
struct strip_inv< Op<T1, op_inv> >
  {
  typedef T1 stored_type;
  
  arma_hot inline
  strip_inv(const Op<T1, op_inv>& X)
    : M(X.m)
    , slow(X.aux_uword_a == 1)
    {
    arma_extra_debug_sigprint();
    }
  
  const T1&  M;
  const bool slow;
  
  static const bool do_inv = true;
  };



template<typename T1>
struct strip_inv< Op<T1, op_inv_sympd> >
  {
  typedef T1 stored_type;
  
  arma_hot inline
  strip_inv(const Op<T1, op_inv_sympd>& X)
    : M(X.m)
    , slow(X.aux_uword_a == 1)
    {
    arma_extra_debug_sigprint();
    }
  
  const T1&  M;
  const bool slow;
  
  static const bool do_inv = true;
  };



//! @}
