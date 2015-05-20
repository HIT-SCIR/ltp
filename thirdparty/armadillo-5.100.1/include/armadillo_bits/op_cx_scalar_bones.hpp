// Copyright (C) 2008-2010 Conrad Sanderson
// Copyright (C) 2008-2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_cx_scalar
//! @{



class op_cx_scalar_times
  {
  public:
  
  template<typename T1>
  inline static void
  apply
    (
          Mat< typename std::complex<typename T1::pod_type> >& out,
    const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_times>& X
    );
  
  template<typename T1>
  inline static void
  apply
    (
             Cube< typename std::complex<typename T1::pod_type> >& out,
    const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_times>& X
    );

  };



class op_cx_scalar_plus
  {
  public:
  
  template<typename T1>
  inline static void
  apply
    (
          Mat< typename std::complex<typename T1::pod_type> >& out,
    const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_plus>& X
    );
  
  template<typename T1>
  inline static void
  apply
    (
             Cube< typename std::complex<typename T1::pod_type> >& out,
    const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_plus>& X
    );

  };



class op_cx_scalar_minus_pre
  {
  public:
  
  template<typename T1>
  inline static void
  apply
    (
          Mat< typename std::complex<typename T1::pod_type> >& out,
    const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_pre>& X
    );
  
  template<typename T1>
  inline static void
  apply
    (
             Cube< typename std::complex<typename T1::pod_type> >& out,
    const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_pre>& X
    );

  };



class op_cx_scalar_minus_post
  {
  public:
  
  template<typename T1>
  inline static void
  apply
    (
          Mat< typename std::complex<typename T1::pod_type> >& out,
    const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_post>& X
    );
  
  template<typename T1>
  inline static void
  apply
    (
             Cube< typename std::complex<typename T1::pod_type> >& out,
    const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_minus_post>& X
    );

  };



class op_cx_scalar_div_pre
  {
  public:
  
  template<typename T1>
  inline static void
  apply
    (
          Mat< typename std::complex<typename T1::pod_type> >& out,
    const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_div_pre>& X
    );
  
  template<typename T1>
  inline static void
  apply
    (
             Cube< typename std::complex<typename T1::pod_type> >& out,
    const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_div_pre>& X
    );

  };



class op_cx_scalar_div_post
  {
  public:
  
  template<typename T1>
  inline static void
  apply
    (
          Mat< typename std::complex<typename T1::pod_type> >& out,
    const mtOp<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_div_post>& X
    );
  
  template<typename T1>
  inline static void
  apply
    (
             Cube< typename std::complex<typename T1::pod_type> >& out,
    const mtOpCube<typename std::complex<typename T1::pod_type>, T1, op_cx_scalar_div_post>& X
    );

  };



//! @}
