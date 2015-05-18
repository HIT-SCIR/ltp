// Copyright (C) 2008-2010 Conrad Sanderson
// Copyright (C) 2008-2010 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup unwrap_cube
//! @{



template<typename T1>
class unwrap_cube
  {
  public:
  
  typedef typename T1::elem_type eT;
  
  inline
  unwrap_cube(const T1& A)
    : M(A)
    {
    arma_extra_debug_sigprint();
    }
  
  const Cube<eT> M;
  };



template<typename eT>
class unwrap_cube< Cube<eT> >
  {
  public:

  inline
  unwrap_cube(const Cube<eT>& A)
    : M(A)
    {
    arma_extra_debug_sigprint();
    }

  const Cube<eT>& M;
  };



//
//
//



template<typename T1>
class unwrap_cube_check
  {
  public:
  
  typedef typename T1::elem_type eT;
  
  inline
  unwrap_cube_check(const T1& A, const Cube<eT>&)
    : M(A)
    {
    arma_extra_debug_sigprint();
    
    arma_type_check(( is_arma_cube_type<T1>::value == false ));
    }
  
  const Cube<eT> M;
  };



template<typename eT>
class unwrap_cube_check< Cube<eT> >
  {
  public:

  inline
  unwrap_cube_check(const Cube<eT>& A, const Cube<eT>& B)
    : M_local( (&A == &B) ? new Cube<eT>(A) : 0 )
    , M      ( (&A == &B) ? (*M_local)      : A )
    {
    arma_extra_debug_sigprint();
    }
  
  
  inline
  ~unwrap_cube_check()
    {
    arma_extra_debug_sigprint();
    
    if(M_local)
      {
      delete M_local;
      }
    }
  
  
  // the order below is important
  const Cube<eT>* M_local;
  const Cube<eT>& M;
  
  };



//! @}
