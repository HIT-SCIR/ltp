// Copyright (C) 2011-2014 Conrad Sanderson
// Copyright (C) 2011-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_symmat
//! @{


template<typename T1>
arma_inline
typename enable_if2< is_cx<typename T1::elem_type>::no, const Op<T1, op_symmat> >::result
symmatu(const Base<typename T1::elem_type,T1>& X, const bool do_conj = false)
  {
  arma_extra_debug_sigprint();
  arma_ignore(do_conj);
  
  return Op<T1, op_symmat>(X.get_ref(), 0, 0);
  }



template<typename T1>
arma_inline
typename enable_if2< is_cx<typename T1::elem_type>::no, const Op<T1, op_symmat> >::result
symmatl(const Base<typename T1::elem_type,T1>& X, const bool do_conj = false)
  {
  arma_extra_debug_sigprint();
  arma_ignore(do_conj);
  
  return Op<T1, op_symmat>(X.get_ref(), 1, 0);
  }



template<typename T1>
arma_inline
typename enable_if2< is_cx<typename T1::elem_type>::yes, const Op<T1, op_symmat_cx> >::result
symmatu(const Base<typename T1::elem_type,T1>& X, const bool do_conj = true)
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_symmat_cx>(X.get_ref(), 0, (do_conj ? 1 : 0));
  }



template<typename T1>
arma_inline
typename enable_if2< is_cx<typename T1::elem_type>::yes, const Op<T1, op_symmat_cx> >::result
symmatl(const Base<typename T1::elem_type,T1>& X, const bool do_conj = true)
  {
  arma_extra_debug_sigprint();
  
  return Op<T1, op_symmat_cx>(X.get_ref(), 1, (do_conj ? 1 : 0));
  }



//! @}
