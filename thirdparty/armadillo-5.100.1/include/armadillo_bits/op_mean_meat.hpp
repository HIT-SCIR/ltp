// Copyright (C) 2009-2012 Conrad Sanderson
// Copyright (C) 2009-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_mean
//! @{



//! \brief
//! For each row or for each column, find the mean value.
//! The result is stored in a dense matrix that has either one column or one row.
//! The dimension, for which the means are found, is set via the mean() function.
template<typename T1>
inline
void
op_mean::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_mean>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap_check<T1> tmp(in.m, out);
  const Mat<eT>& X = tmp.M;
  
  const uword dim = in.aux_uword_a;
  arma_debug_check( (dim > 1), "mean(): incorrect usage. dim must be 0 or 1");
  
  const uword X_n_rows = X.n_rows;
  const uword X_n_cols = X.n_cols;
  
  if(dim == 0)
    {
    arma_extra_debug_print("op_mean::apply(), dim = 0");
    
    out.set_size( (X_n_rows > 0) ? 1 : 0, X_n_cols );
    
    if(X_n_rows > 0)
      {
      eT* out_mem = out.memptr();
      
      for(uword col=0; col < X_n_cols; ++col)
        {
        out_mem[col] = op_mean::direct_mean( X.colptr(col), X_n_rows );
        }
      }
    }
  else
  if(dim == 1)
    {
    arma_extra_debug_print("op_mean::apply(), dim = 1");
    
    out.set_size(X_n_rows, (X_n_cols > 0) ? 1 : 0);
    
    if(X_n_cols > 0)
      {
      eT* out_mem = out.memptr();
      
      for(uword row=0; row < X_n_rows; ++row)
        {
        out_mem[row] = op_mean::direct_mean( X, row );
        }
      }
    }
  }



template<typename eT>
arma_pure
inline
eT
op_mean::direct_mean(const eT* const X, const uword n_elem)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  const eT result = arrayops::accumulate(X, n_elem) / T(n_elem);
  
  return arma_isfinite(result) ? result : op_mean::direct_mean_robust(X, n_elem);
  }



template<typename eT>
arma_pure
inline
eT
op_mean::direct_mean_robust(const eT* const X, const uword n_elem)
  {
  arma_extra_debug_sigprint();
  
  // use an adapted form of the mean finding algorithm from the running_stat class
  
  typedef typename get_pod_type<eT>::result T;
  
  uword i,j;
  
  eT r_mean = eT(0);
  
  for(i=0, j=1; j<n_elem; i+=2, j+=2)
    {
    const eT Xi = X[i];
    const eT Xj = X[j];
    
    r_mean = r_mean + (Xi - r_mean)/T(j);    // we need i+1, and j is equivalent to i+1 here
    r_mean = r_mean + (Xj - r_mean)/T(j+1);
    }
  
  
  if(i < n_elem)
    {
    const eT Xi = X[i];
    
    r_mean = r_mean + (Xi - r_mean)/T(i+1);
    }
  
  return r_mean;
  }



template<typename eT>
inline
eT
op_mean::direct_mean(const Mat<eT>& X, const uword row)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  const uword X_n_cols = X.n_cols;
  
  eT val = eT(0);
  
  uword i,j;
  for(i=0, j=1; j < X_n_cols; i+=2, j+=2)
    {
    val += X.at(row,i);
    val += X.at(row,j);
    }
  
  if(i < X_n_cols)
    {
    val += X.at(row,i);
    }
  
  const eT result = val / T(X_n_cols);
  
  return arma_isfinite(result) ? result : op_mean::direct_mean_robust(X, row);
  }



template<typename eT>
inline
eT
op_mean::direct_mean_robust(const Mat<eT>& X, const uword row)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  const uword X_n_cols = X.n_cols;
  
  eT r_mean = eT(0);
  
  for(uword col=0; col < X_n_cols; ++col)
    {
    r_mean = r_mean + (X.at(row,col) - r_mean)/T(col+1);
    }
  
  return r_mean;
  }



template<typename eT>
inline
eT
op_mean::mean_all(const subview<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  const uword X_n_rows = X.n_rows;
  const uword X_n_cols = X.n_cols;
  const uword X_n_elem = X.n_elem;
  
  arma_debug_check( (X_n_elem == 0), "mean(): given object has no elements" );
  
  eT val = eT(0);
  
  if(X_n_rows == 1)
    {
    const Mat<eT>& A = X.m;
    
    const uword start_row = X.aux_row1;
    const uword start_col = X.aux_col1;
    
    const uword end_col_p1 = start_col + X_n_cols;
    
    uword i,j;
    for(i=start_col, j=start_col+1; j < end_col_p1; i+=2, j+=2)
      {
      val += A.at(start_row, i);
      val += A.at(start_row, j);
      }
    
    if(i < end_col_p1)
      {
      val += A.at(start_row, i);
      }
    }
  else
    {
    for(uword col=0; col < X_n_cols; ++col)
      {
      val += arrayops::accumulate(X.colptr(col), X_n_rows);
      }
    }
  
  const eT result = val / T(X_n_elem);
  
  return arma_isfinite(result) ? result : op_mean::mean_all_robust(X);
  }



template<typename eT>
inline 
eT
op_mean::mean_all_robust(const subview<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  const uword X_n_rows = X.n_rows;
  const uword X_n_cols = X.n_cols;
  
  const uword start_row = X.aux_row1;
  const uword start_col = X.aux_col1;
  
  const uword end_row_p1 = start_row + X_n_rows;
  const uword end_col_p1 = start_col + X_n_cols;
  
  const Mat<eT>& A = X.m;
  
  
  eT r_mean = eT(0);
  
  if(X_n_rows == 1)
    {
    uword i=0;
    
    for(uword col = start_col; col < end_col_p1; ++col, ++i)
      {
      r_mean = r_mean + (A.at(start_row,col) - r_mean)/T(i+1);
      }
    }
  else
    {
    uword i=0;
    
    for(uword col = start_col; col < end_col_p1; ++col)
    for(uword row = start_row; row < end_row_p1; ++row, ++i)
      {
      r_mean = r_mean + (A.at(row,col) - r_mean)/T(i+1);
      }
    }
  
  return r_mean;
  }



template<typename eT>
inline 
eT
op_mean::mean_all(const diagview<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  const uword X_n_elem = X.n_elem;
  
  arma_debug_check( (X_n_elem == 0), "mean(): given object has no elements" );
  
  eT val = eT(0);
  
  for(uword i=0; i<X_n_elem; ++i)
    {
    val += X[i];
    }
  
  const eT result = val / T(X_n_elem);
  
  return arma_isfinite(result) ? result : op_mean::mean_all_robust(X);
  }



template<typename eT>
inline 
eT
op_mean::mean_all_robust(const diagview<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  const uword X_n_elem = X.n_elem;
  
  eT r_mean = eT(0);
  
  for(uword i=0; i<X_n_elem; ++i)
    {
    r_mean = r_mean + (X[i] - r_mean)/T(i+1);
    }
  
  return r_mean;
  }



template<typename T1>
inline
typename T1::elem_type 
op_mean::mean_all(const Base<typename T1::elem_type, T1>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap<T1>   tmp(X.get_ref());
  const Mat<eT>& A = tmp.M;
  
  const uword A_n_elem = A.n_elem;
  
  arma_debug_check( (A_n_elem == 0), "mean(): given object has no elements" );
  
  return op_mean::direct_mean(A.memptr(), A_n_elem);
  }



template<typename eT>
arma_inline
eT
op_mean::robust_mean(const eT A, const eT B)
  {
  return A + (B - A)/eT(2);
  }



template<typename T>
arma_inline
std::complex<T>
op_mean::robust_mean(const std::complex<T>& A, const std::complex<T>& B)
  {
  return A + (B - A)/T(2);
  }



//! @}

