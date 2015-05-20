// Copyright (C) 2009-2013 Conrad Sanderson
// Copyright (C) 2009-2013 NICTA (www.nicta.com.au)
// Copyright (C) 2009-2010 Dimitrios Bouzas
// Copyright (C) 2011 Stanislav Funiak
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_rank
//! @{



template<typename T1>
inline
arma_warn_unused
uword
rank
  (
  const Base<typename T1::elem_type,T1>& X,
        typename T1::pod_type            tol = 0.0,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::pod_type T;
  
  uword  X_n_rows;
  uword  X_n_cols;
  Col<T> s;
  
  const bool  status = auxlib::svd_dc(s, X, X_n_rows, X_n_cols);
  const uword n_elem = s.n_elem;
  
  if(status == true)
    {
    if( (tol == T(0)) && (n_elem > 0) )
      {
      tol = (std::max)(X_n_rows, X_n_cols) * eop_aux::direct_eps(max(s));
      }
    
    // count non zero valued elements in s
    
    const T* s_mem = s.memptr();
    
    uword count = 0;
    
    for(uword i=0; i<n_elem; ++i)
      {
      if(s_mem[i] > tol) { ++count; }
      }
    
    return count;
    }
  else
    {
    arma_bad("rank(): failed to converge");
    
    return uword(0);
    }
  }



//! @}
