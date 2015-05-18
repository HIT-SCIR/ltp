// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup operator_ostream
//! @{



template<typename eT, typename T1>
inline
std::ostream&
operator<< (std::ostream& o, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  const unwrap<T1> tmp(X.get_ref());
  
  arma_ostream::print(o, tmp.M, true);
  
  return o;
  }



template<typename eT, typename T1>
inline
std::ostream&
operator<< (std::ostream& o, const SpBase<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  const unwrap_spmat<T1> tmp(X.get_ref());
  
  arma_ostream::print(o, tmp.M, true);
  
  return o;
  }



template<typename T1>
inline
std::ostream&
operator<< (std::ostream& o, const SpValProxy<T1>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  o << eT(X);
  
  return o;
  }



template<typename T1>
inline
std::ostream&
operator<< (std::ostream& o, const BaseCube<typename T1::elem_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  const unwrap_cube<T1> tmp(X.get_ref());
  
  arma_ostream::print(o, tmp.M, true);
  
  return o;
  }



//! Print the contents of a field to the specified stream.
template<typename T1>
inline
std::ostream&
operator<< (std::ostream& o, const field<T1>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_ostream::print(o, X);
  
  return o;
  }



//! Print the contents of a subfield to the specified stream
template<typename T1>
inline
std::ostream&
operator<< (std::ostream& o, const subview_field<T1>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_ostream::print(o, X);

  return o;
  }



inline
std::ostream&
operator<< (std::ostream& o, const SizeMat& S)
  {
  arma_extra_debug_sigprint();
  
  arma_ostream::print(o, S);
  
  return o;
  }



inline
std::ostream&
operator<< (std::ostream& o, const SizeCube& S)
  {
  arma_extra_debug_sigprint();
  
  arma_ostream::print(o, S);
  
  return o;
  }



//! @}
