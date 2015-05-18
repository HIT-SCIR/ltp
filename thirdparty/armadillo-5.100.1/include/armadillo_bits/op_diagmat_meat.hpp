// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_diagmat
//! @{



template<typename T1>
inline
void
op_diagmat::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_diagmat>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const Proxy<T1> P(X.m);
  
  const uword n_rows = P.get_n_rows();
  const uword n_cols = P.get_n_cols();
  
  const bool P_is_vec = (n_rows == 1) || (n_cols == 1);
  
  
  if(P.is_alias(out) == false)
    {
    if(P_is_vec)    // generate a diagonal matrix out of a vector
      {
      const uword N = (n_rows == 1) ? n_cols : n_rows;
      
      out.zeros(N, N);
      
      if(Proxy<T1>::prefer_at_accessor == false)
        {
        typename Proxy<T1>::ea_type P_ea = P.get_ea();
        
        for(uword i=0; i < N; ++i) { out.at(i,i) = P_ea[i]; }
        }
      else
        {
        if(n_rows == 1)
          {
          for(uword i=0; i < N; ++i) { out.at(i,i) = P.at(0,i); }
          }
        else
          {
          for(uword i=0; i < N; ++i) { out.at(i,i) = P.at(i,0); }
          }
        }
      }
    else   // generate a diagonal matrix out of a matrix
      {
      arma_debug_check( (n_rows != n_cols), "diagmat(): given matrix is not square" );
      
      out.zeros(n_rows, n_rows);
      
      for(uword i=0; i < n_rows; ++i) { out.at(i,i) = P.at(i,i); }
      }
    }
  else   // we have aliasing
    {
    if(P_is_vec)   // generate a diagonal matrix out of a vector
      {
      const uword N = (n_rows == 1) ? n_cols : n_rows;
      
      podarray<eT> tmp(N);
      eT* tmp_mem = tmp.memptr();
      
      if(Proxy<T1>::prefer_at_accessor == false)
        {
        typename Proxy<T1>::ea_type P_ea = P.get_ea();
        
        for(uword i=0; i < N; ++i) { tmp_mem[i] = P_ea[i]; }
        }
      else
        {
        if(n_rows == 1)
          {
          for(uword i=0; i < N; ++i) { tmp_mem[i] = P.at(0,i); }
          }
        else
          {
          for(uword i=0; i < N; ++i) { tmp_mem[i] = P.at(i,0); }
          }
        }
      
      out.zeros(N, N);
      
      for(uword i=0; i < N; ++i) { out.at(i,i) = tmp_mem[i]; }
      }
    else   // generate a diagonal matrix out of a matrix
      {
      arma_debug_check( (n_rows != n_cols), "diagmat(): given matrix is not square" );
      
      if( (Proxy<T1>::has_subview == false) && (Proxy<T1>::fake_mat == false) )
        {
        // NOTE: we have aliasing and it's not due to a subview, hence we're assuming that the output matrix already has the correct size
        
        for(uword i=0; i < n_rows; ++i)
          {
          const eT val = P.at(i,i);
          
          arrayops::fill_zeros(out.colptr(i), n_rows);
          
          out.at(i,i) = val;
          }
        }
      else
        {
        podarray<eT> tmp(n_rows);
        eT* tmp_mem = tmp.memptr();
        
        for(uword i=0; i < n_rows; ++i)  { tmp_mem[i] = P.at(i,i); }
        
        out.zeros(n_rows, n_rows);
        
        for(uword i=0; i < n_rows; ++i)  { out.at(i,i) = tmp_mem[i]; }
        }
      }
    }
  }



//! @}
