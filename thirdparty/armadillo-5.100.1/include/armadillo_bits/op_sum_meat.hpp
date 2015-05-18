// Copyright (C) 2008-2015 Conrad Sanderson
// Copyright (C) 2008-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_sum
//! @{

//! \brief
//! Immediate sum of elements of a matrix along a specified dimension (either rows or columns).
//! The result is stored in a dense matrix that has either one column or one row.
//! See the sum() function for more details.
template<typename T1>
arma_hot
inline
void
op_sum::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_sum>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword dim = in.aux_uword_a;
  arma_debug_check( (dim > 1), "sum(): incorrect usage. dim must be 0 or 1");
  
  const Proxy<T1> P(in.m);
  
  typedef typename Proxy<T1>::stored_type P_stored_type;
  
  const bool is_alias = P.is_alias(out);
  
  if( (is_Mat<P_stored_type>::value == true) || is_alias )
    {
    const unwrap_check<P_stored_type> tmp(P.Q, is_alias);
    
    const typename unwrap_check<P_stored_type>::stored_type& X = tmp.M;
    
    const uword X_n_rows = X.n_rows;
    const uword X_n_cols = X.n_cols;
    
    if(dim == 0)  // traverse across rows (i.e. find the sum in each column)
      {
      out.set_size(1, X_n_cols);
      
      eT* out_mem = out.memptr();
      
      for(uword col=0; col < X_n_cols; ++col)
        {
        out_mem[col] = arrayops::accumulate( X.colptr(col), X_n_rows );
        }
      }
    else  // traverse across columns (i.e. find the sum in each row)
      {
      out.set_size(X_n_rows, 1);
      
      eT* out_mem = out.memptr();
        
      for(uword row=0; row < X_n_rows; ++row)
        {
        eT val1 = eT(0);
        eT val2 = eT(0);
        
        const eT* rowptr = &(X.at(row,0));
        
        uword j;
        for(j=1; j < X_n_cols; j+=2)
          {
          val1 += (*rowptr);  rowptr += X_n_rows;
          val2 += (*rowptr);  rowptr += X_n_rows;
          }
        
        if((j-1) < X_n_cols)
          {
          val1 += (*rowptr);
          }
        
        out_mem[row] = (val1 + val2);
        }
      }
    }
  else
    {
    const uword P_n_rows = P.get_n_rows();
    const uword P_n_cols = P.get_n_cols();
    
    if(dim == 0)  // traverse across rows (i.e. find the sum in each column)
      {
      out.set_size(1, P_n_cols);
      
      eT* out_mem = out.memptr();
      
      for(uword col=0; col < P_n_cols; ++col)
        {
        eT val1 = eT(0);
        eT val2 = eT(0);
        
        uword i,j;
        for(i=0, j=1; j < P_n_rows; i+=2, j+=2)
          {
          val1 += P.at(i,col);
          val2 += P.at(j,col);
          }
        
        if(i < P_n_rows)
          {
          val1 += P.at(i,col);
          }
        
        out_mem[col] = (val1 + val2);
        }
      }
    else  // traverse across columns (i.e. find the sum in each row)
      {
      out.set_size(P_n_rows, 1);
      
      eT* out_mem = out.memptr();
      
      for(uword row=0; row < P_n_rows; ++row)
        {
        eT val1 = eT(0);
        eT val2 = eT(0);
        
        uword i,j;
        for(i=0, j=1; j < P_n_cols; i+=2, j+=2)
          {
          val1 += P.at(row,i);
          val2 += P.at(row,j);
          }
        
        if(i < P_n_cols)
          {
          val1 += P.at(row,i);
          }
        
        out_mem[row] = (val1 + val2);
        }
      }
    }
  }



//! @}
