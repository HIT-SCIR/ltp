// Copyright (C) 2013 Conrad Sanderson
// Copyright (C) 2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup op_fft
//! @{



class op_fft_real
  {
  public:
  
  template<typename T1>
  inline static void apply( Mat< std::complex<typename T1::pod_type> >& out, const mtOp<std::complex<typename T1::pod_type>,T1,op_fft_real>& in );
  };



class op_fft_cx
  {
  public:
  
  template<typename T1>
  inline static void apply( Mat<typename T1::elem_type>& out, const Op<T1,op_fft_cx>& in );
  
  template<typename T1, bool inverse>
  inline static void apply_noalias(Mat<typename T1::elem_type>& out, const Proxy<T1>& P, const uword a, const uword b);

  template<typename T1> arma_hot inline static void copy_vec       (typename Proxy<T1>::elem_type* dest, const Proxy<T1>& P, const uword N);
  template<typename T1> arma_hot inline static void copy_vec_proxy (typename Proxy<T1>::elem_type* dest, const Proxy<T1>& P, const uword N);
  template<typename T1> arma_hot inline static void copy_vec_unwrap(typename Proxy<T1>::elem_type* dest, const Proxy<T1>& P, const uword N);
  };



class op_ifft_cx
  {
  public:
  
  template<typename T1>
  inline static void apply( Mat<typename T1::elem_type>& out, const Op<T1,op_ifft_cx>& in );
  };



//! @}
