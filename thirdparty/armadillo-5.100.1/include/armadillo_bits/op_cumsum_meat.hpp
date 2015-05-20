// Copyright (C) 2010-2015 Conrad Sanderson
// Copyright (C) 2010-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_cumsum
//! @{


template<typename eT>
inline
void
op_cumsum_mat::apply_noalias(Mat<eT>& out, const Mat<eT>& X, const uword dim)
  {
  arma_extra_debug_sigprint();
  
  out.copy_size(X);
  
  const uword X_n_rows = X.n_rows;
  const uword X_n_cols = X.n_cols;
  
  if(dim == 0)
    {
    arma_extra_debug_print("op_cumsum_mat::apply(), dim = 0");
    
    for(uword col=0; col<X_n_cols; ++col)
      {
            eT* out_colmem = out.colptr(col);
      const eT* X_colmem   = X.colptr(col);
      
      eT acc = eT(0);
      
      for(uword row=0; row<X_n_rows; ++row)
        {
        acc += X_colmem[row];
        
        out_colmem[row] = acc;
        }
      }
    }
  else
  if(dim == 1)
    {
    arma_extra_debug_print("op_cumsum_mat::apply(), dim = 1");
    
    for(uword row=0; row<X_n_rows; ++row)
      {
      eT acc = eT(0);
      
      for(uword col=0; col<X_n_cols; ++col)
        {
        acc += X.at(row,col);
        
        out.at(row,col) = acc;
        }
      }
    }
  }



template<typename T1>
inline
void
op_cumsum_mat::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_cumsum_mat>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap<T1>   tmp(in.m);
  const Mat<eT>& X = tmp.M;
  
  const uword dim = in.aux_uword_a;
  arma_debug_check( (dim > 1), "cumsum(): incorrect usage. dim must be 0 or 1");
  
  if(&out == &X)
    {
    Mat<eT> out2;
    
    op_cumsum_mat::apply_noalias(out2, X, dim);
    
    out.steal_mem(out2);
    }
  else
    {
    op_cumsum_mat::apply_noalias(out, X, dim);
    }
  }



template<typename eT>
inline
void
op_cumsum_vec::apply_noalias(Mat<eT>& out, const Mat<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  const uword n_elem = X.n_elem;
  
  out.copy_size(X);
  
        eT* out_mem = out.memptr();
  const eT* X_mem   = X.memptr();
  
  eT acc = eT(0);
  
  for(uword i=0; i<n_elem; ++i)
    {
    acc += X_mem[i];
    
    out_mem[i] = acc;
    }
  }




template<typename T1>
inline
void
op_cumsum_vec::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_cumsum_vec>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const quasi_unwrap<T1> U(in.m);
  
  const Mat<eT>& X = U.M;
  
  if(U.is_alias(out))
    {
    Mat<eT> out2;
    
    op_cumsum_vec::apply_noalias(out2, X);
    
    out.steal_mem(out2);
    }
  else
    {
    op_cumsum_vec::apply_noalias(out, X);
    }
  }



//! @}

