// Copyright (C) 2015 Ryan Curtin
// Copyright (C) 2015 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_spsolve
//! @{

//! Solve a system of linear equations, i.e., A*X = B, where X is unknown,
//! A is sparse, and B is dense.  X will be dense too.

template<typename T1, typename T2>
inline
bool
spsolve_helper
  (
           Mat<typename T1::elem_type>&     out,
  const SpBase<typename T1::elem_type, T1>& A,
  const   Base<typename T1::elem_type, T2>& B,
  const char*                          solver,
  const spsolve_opts_base&             settings,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  
  const char sig = (solver != NULL) ? solver[0] : char(0);
  
  arma_debug_check( ((sig != 'l') && (sig != 's')), "spsolve(): unknown solver" );
  
  bool status = false;
  
  if(sig == 's')  // SuperLU solver
    {
    const superlu_opts& opts = (settings.id == 1) ? static_cast<const superlu_opts&>(settings) : superlu_opts();
    
    arma_debug_check( ( (opts.pivot_thresh < double(0)) || (opts.pivot_thresh > double(1)) ), "spsolve(): pivot_thresh out of bounds" );
    
    status = sp_auxlib::spsolve(out, A.get_ref(), B.get_ref(), opts);
    }
  else
  if(sig == 'l')  // brutal LAPACK solver
    {
    arma_debug_warn( (settings.id != 0), "spsolve(): ignoring settings not applicable to LAPACK based solver" );
    
    Mat<eT> AA;
    
    bool conversion_ok = true;
    
    try
      {
      Mat<eT> tmp(A.get_ref());  // conversion from sparse to dense can throw std::bad_alloc
      
      AA.steal_mem(tmp);
      }
    catch(std::bad_alloc&)
      {
      conversion_ok = false;
      
      arma_debug_warn(true, "spsolve(): not enough memory to use LAPACK based solver");
      }
    
    if(conversion_ok)
      {
      arma_debug_check( (AA.n_rows != AA.n_cols), "spsolve(): matrix A must be square sized" );
      
      status = auxlib::solve(out, AA, B.get_ref(), true);
      }
    }
  
  if(status == false)  { out.reset(); }
  
  return status;
  }



template<typename T1, typename T2>
inline
bool
spsolve
  (
           Mat<typename T1::elem_type>&     out,
  const SpBase<typename T1::elem_type, T1>& A,
  const   Base<typename T1::elem_type, T2>& B,
  const char*                          solver   = "superlu",
  const spsolve_opts_base&             settings = spsolve_opts_none(),
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const bool status = spsolve_helper(out, A.get_ref(), B.get_ref(), solver, settings);
  
  if(status == false)
    {
    arma_debug_warn(true, "spsolve(): solution not found");
    }
  
  return status;
  }



template<typename T1, typename T2>
inline
Mat<typename T1::elem_type>
spsolve
  (
  const SpBase<typename T1::elem_type, T1>& A,
  const   Base<typename T1::elem_type, T2>& B,
  const char*                          solver   = "superlu",
  const spsolve_opts_base&             settings = spsolve_opts_none(),
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  
  Mat<eT> out;
  
  const bool status = spsolve_helper(out, A.get_ref(), B.get_ref(), solver, settings);
  
  if(status == false)
    {
    arma_bad("spsolve(): solution not found");
    }
  
  return out;
  }



//! @}
