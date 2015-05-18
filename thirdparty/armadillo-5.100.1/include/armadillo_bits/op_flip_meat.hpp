// Copyright (C) 2010-2015 Conrad Sanderson
// Copyright (C) 2010-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_flip
//! @{



template<typename T1>
inline
void
op_flipud::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_flipud>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap<T1>   tmp(in.m);
  const Mat<eT>& X = tmp.M;
  
  const uword X_n_rows = X.n_rows;
  
  if(&out != &X)
    {
    out.copy_size(X);
    
    if(T1::is_col || X.is_colvec())
      {
      for(uword i=0; i<X_n_rows; ++i)  { out[i] = X[X_n_rows-1 - i]; }
      }
    else
      {
      for(uword i=0; i<X_n_rows; ++i)  { out.row(i) = X.row(X_n_rows-1 - i); }
      }
    }
  else
    {
    const uword N = X_n_rows / 2;
    
    if(T1::is_col || X.is_colvec())
      {
      for(uword i=0; i<N; ++i)  { std::swap(out[i], out[X_n_rows-1 - i]); }
      }
    else
      {
      for(uword i=0; i<N; ++i)  { out.swap_rows(i, X_n_rows-1 - i); }
      }
    }
  }



template<typename T1>
inline
void
op_fliplr::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_fliplr>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap<T1>   tmp(in.m);
  const Mat<eT>& X = tmp.M;
  
  const uword X_n_cols = X.n_cols;
  
  if(&out != &X)
    {
    out.copy_size(X);
    
    if(T1::is_row || X.is_rowvec())
      {
      for(uword i=0; i<X_n_cols; ++i)  { out[i] = X[X_n_cols-1 - i]; }
      }
    else
      {
      for(uword i=0; i<X_n_cols; ++i)  { out.col(i) = X.col(X_n_cols-1 - i); }
      }
    }
  else
    {
    const uword N = X_n_cols / 2;
    
    if(T1::is_row || X.is_rowvec())
      {
      for(uword i=0; i<N; ++i)  { std::swap(out[i], out[X_n_cols-1 - i]); }
      }
    else
      {
      for(uword i=0; i<N; ++i)  { out.swap_cols(i, X_n_cols-1 - i); }
      }
    }
  }



//! @}
