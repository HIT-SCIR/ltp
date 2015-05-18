// Copyright (C) 2013 Conrad Sanderson
// Copyright (C) 2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_fft2
//! @{



// 2D FFT & 2D IFFT



template<typename T1>
inline
typename
enable_if2
  <
  is_arma_type<T1>::value,
  Mat< std::complex<typename T1::pod_type> >
  >::result
fft2(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  // not exactly efficient, but "better-than-nothing" implementation
  
  typedef typename T1::pod_type T;
  
  Mat< std::complex<T> > B = fft(A);
  
  // for square matrices, strans() will work out that an inplace transpose can be done,
  // hence we can potentially avoid creating a temporary matrix
  
  B = strans(B);
  
  return strans( fft(B) );
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_arma_type<T1>::value,
  Mat< std::complex<typename T1::pod_type> >
  >::result
fft2(const T1& A, const uword n_rows, const uword n_cols)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap<T1>   tmp(A);
  const Mat<eT>& B = tmp.M;
  
  const bool do_resize = (B.n_rows != n_rows) || (B.n_cols != n_cols);
  
  return fft2( do_resize ? resize(B,n_rows,n_cols) : B );
  }



template<typename T1>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_complex_strict<typename T1::elem_type>::value),
  Mat< std::complex<typename T1::pod_type> >
  >::result
ifft2(const T1& A)
  {
  arma_extra_debug_sigprint();
  
  // not exactly efficient, but "better-than-nothing" implementation
  
  typedef typename T1::pod_type T;
  
  Mat< std::complex<T> > B = ifft(A);
  
  // for square matrices, strans() will work out that an inplace transpose can be done,
  // hence we can potentially avoid creating a temporary matrix
  
  B = strans(B);
  
  return strans( ifft(B) );
  }



template<typename T1>
inline
typename
enable_if2
  <
  (is_arma_type<T1>::value && is_complex_strict<typename T1::elem_type>::value),
  Mat< std::complex<typename T1::pod_type> >
  >::result
ifft2(const T1& A, const uword n_rows, const uword n_cols)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap<T1>   tmp(A);
  const Mat<eT>& B = tmp.M;
  
  const bool do_resize = (B.n_rows != n_rows) || (B.n_cols != n_cols);
  
  return ifft2( do_resize ? resize(B,n_rows,n_cols) : B );
  }



//! @}
