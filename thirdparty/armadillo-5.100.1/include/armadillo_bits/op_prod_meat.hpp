// Copyright (C) 2009-2012 Conrad Sanderson
// Copyright (C) 2009-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_prod
//! @{

//! \brief
//! Immediate product of elements of a matrix along a specified dimension (either rows or columns).
//! The result is stored in a dense matrix that has either one column or one row.
//! See the prod() function for more details.
template<typename T1>
inline
void
op_prod::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_prod>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword dim = in.aux_uword_a;
  arma_debug_check( (dim > 1), "prod(): incorrect usage. dim must be 0 or 1");
  
  const unwrap_check<T1> tmp(in.m, out);
  const Mat<eT>& X     = tmp.M;
  
  const uword X_n_rows = X.n_rows;
  const uword X_n_cols = X.n_cols;
    
  if(dim == 0)  // traverse across rows (i.e. find the product in each column)
    {
    out.set_size(1, X_n_cols);
    
    eT* out_mem = out.memptr();
    
    for(uword col=0; col<X_n_cols; ++col)
      {
      out_mem[col] = arrayops::product(X.colptr(col), X_n_rows);
      }
    }
  else  // traverse across columns (i.e. find the product in each row)
    {
    out.set_size(X_n_rows, 1);
    
    eT* out_mem = out.memptr();
    
    for(uword row=0; row<X_n_rows; ++row)
      {
      eT val = eT(1);
      
      uword i,j;
      for(i=0, j=1; j < X_n_cols; i+=2, j+=2)
        {
        val *= X.at(row,i);
        val *= X.at(row,j);
        }
      
      if(i < X_n_cols)
        {
        val *= X.at(row,i);
        }
      
      out_mem[row] = val;
      }
    }
  }



template<typename eT>
inline
eT
op_prod::prod(const subview<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  eT val = eT(1);
  
  const uword X_n_rows = X.n_rows;
  const uword X_n_cols = X.n_cols;
  
  if(X_n_rows == 1)
    {
    const Mat<eT>& A = X.m;
  
    const uword start_row = X.aux_row1;
    const uword start_col = X.aux_col1;
    
    const uword end_col_p1 = start_col + X_n_cols;
    
    uword i,j;
    for(i=start_col, j=start_col+1; j < end_col_p1; i+=2, j+=2)
      {
      val *= A.at(start_row, i);
      val *= A.at(start_row, j);
      }
    
    if(i < end_col_p1)
      {
      val *= A.at(start_row, i);
      }
    }
  else
    {
    for(uword col=0; col < X_n_cols; ++col)
      {
      val *= arrayops::product( X.colptr(col), X_n_rows );
      }
    }
  
  return val;
  }



template<typename T1>
inline
typename T1::elem_type
op_prod::prod(const Base<typename T1::elem_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const Proxy<T1> P(X.get_ref());
  
  eT val = eT(1);
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typedef typename Proxy<T1>::ea_type ea_type;
    
    const ea_type A = P.get_ea();
    
    const uword n_elem = P.get_n_elem();
    
    uword i,j;
    for(i=0, j=1; j < n_elem; i+=2, j+=2)
      {
      val *= A[i];
      val *= A[j];
      }
    
    if(i < n_elem)
      {
      val *= A[i];
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    if(n_rows == 1)
      {
      uword i,j;
      for(i=0, j=1; j < n_cols; i+=2, j+=2)
        {
        val *= P.at(0,i);
        val *= P.at(0,j);
        }
      
      if(i < n_cols)
        {
        val *= P.at(0,i);
        }
      }
    else
      {
      for(uword col=0; col < n_cols; ++col)
        {
        uword i,j;
        for(i=0, j=1; j < n_rows; i+=2, j+=2)
          {
          val *= P.at(i,col);
          val *= P.at(j,col);
          }
        
        if(i < n_rows)
          {
          val *= P.at(i,col);
          }
        }
      }
    }
  
  return val;
  }



//! @}
