// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_max
//! @{



//! \brief
//! For each row or for each column, find the maximum value.
//! The result is stored in a dense matrix that has either one column or one row.
//! The dimension, for which the maxima are found, is set via the max() function.
template<typename T1>
inline
void
op_max::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_max>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap_check<T1> tmp(in.m, out);
  const Mat<eT>& X     = tmp.M;
  
  const uword dim = in.aux_uword_a;
  arma_debug_check( (dim > 1), "max(): incorrect usage. dim must be 0 or 1");
  
  const uword X_n_rows = X.n_rows;
  const uword X_n_cols = X.n_cols;
  
  if(dim == 0)
    {
    arma_extra_debug_print("op_max::apply(), dim = 0");
    
    arma_debug_check( (X_n_rows == 0), "max(): given object has zero rows" );

    out.set_size(1, X_n_cols);
    
    eT* out_mem = out.memptr();
    
    for(uword col=0; col<X_n_cols; ++col)
      {
      out_mem[col] = op_max::direct_max( X.colptr(col), X_n_rows );
      }
    }
  else
  if(dim == 1)
    {
    arma_extra_debug_print("op_max::apply(), dim = 1");
    
    arma_debug_check( (X_n_cols == 0), "max(): given object has zero columns" );

    out.set_size(X_n_rows, 1);
    
    eT* out_mem = out.memptr();
    
    for(uword row=0; row<X_n_rows; ++row)
      {
      out_mem[row] = op_max::direct_max( X, row );
      }
    }
  }



template<typename eT>
arma_pure
inline
eT
op_max::direct_max(const eT* const X, const uword n_elem)
  {
  arma_extra_debug_sigprint();
  
  eT max_val = priv::most_neg<eT>();
  
  uword i,j;
  
  for(i=0, j=1; j<n_elem; i+=2, j+=2)
    {
    const eT X_i = X[i];
    const eT X_j = X[j];
    
    if(X_i > max_val) { max_val = X_i; }
    if(X_j > max_val) { max_val = X_j; }
    }
  
  
  if(i < n_elem)
    {
    const eT X_i = X[i];
    
    if(X_i > max_val) { max_val = X_i; }
    }
  
  return max_val;
  }



template<typename eT>
inline
eT
op_max::direct_max(const eT* const X, const uword n_elem, uword& index_of_max_val)
  {
  arma_extra_debug_sigprint();
  
  eT max_val = priv::most_neg<eT>();
  
  uword best_index = 0;
  
  uword i,j;
  
  for(i=0, j=1; j<n_elem; i+=2, j+=2)
    {
    const eT X_i = X[i];
    const eT X_j = X[j];
    
    if(X_i > max_val)
      {
      max_val    = X_i;
      best_index = i;
      }
    
    if(X_j > max_val)
      {
      max_val    = X_j;
      best_index = j;
      }
    }
  
  
  if(i < n_elem)
    {
    const eT X_i = X[i];
    
    if(X_i > max_val)
      {
      max_val    = X_i;
      best_index = i;
      }
    }
  
  index_of_max_val = best_index;
  
  return max_val;
  }



template<typename eT>
inline
eT
op_max::direct_max(const Mat<eT>& X, const uword row)
  {
  arma_extra_debug_sigprint();
  
  const uword X_n_cols = X.n_cols;
  
  eT max_val = priv::most_neg<eT>();
  
  uword i,j;
  for(i=0, j=1; j < X_n_cols; i+=2, j+=2)
    {
    const eT tmp_i = X.at(row,i);
    const eT tmp_j = X.at(row,j);
    
    if(tmp_i > max_val) { max_val = tmp_i; }
    if(tmp_j > max_val) { max_val = tmp_j; }
    }
  
  if(i < X_n_cols)
    {
    const eT tmp_i = X.at(row,i);
    
    if(tmp_i > max_val) { max_val = tmp_i; }
    }
  
  return max_val;
  }



template<typename eT>
inline
eT
op_max::max(const subview<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (X.n_elem == 0), "max(): given object has no elements" );

  const uword X_n_rows = X.n_rows;
  const uword X_n_cols = X.n_cols;
  
  eT max_val = priv::most_neg<eT>();
  
  if(X_n_rows == 1)
    {
    const Mat<eT>& A = X.m;
    
    const uword start_row = X.aux_row1;
    const uword start_col = X.aux_col1;
    
    const uword end_col_p1 = start_col + X_n_cols;
  
    uword i,j;
    for(i=start_col, j=start_col+1; j < end_col_p1; i+=2, j+=2)
      {
      const eT tmp_i = A.at(start_row, i);
      const eT tmp_j = A.at(start_row, j);
      
      if(tmp_i > max_val) { max_val = tmp_i; }
      if(tmp_j > max_val) { max_val = tmp_j; }
      }
    
    if(i < end_col_p1)
      {
      const eT tmp_i = A.at(start_row, i);
      
      if(tmp_i > max_val) { max_val = tmp_i; }
      }
    }
  else
    {
    for(uword col=0; col < X_n_cols; ++col)
      {
      eT tmp_val = op_max::direct_max(X.colptr(col), X_n_rows);
      
      if(tmp_val > max_val) { max_val = tmp_val; }
      }
    }
  
  return max_val;
  }



template<typename T1>
inline
typename arma_not_cx<typename T1::elem_type>::result
op_max::max(const Base<typename T1::elem_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const Proxy<T1> P(X.get_ref());
  
  const uword n_elem = P.get_n_elem();
  
  arma_debug_check( (n_elem == 0), "max(): given object has no elements" );
  
  eT max_val = priv::most_neg<eT>();
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typedef typename Proxy<T1>::ea_type ea_type;
    
    ea_type A = P.get_ea();
    
    uword i,j;
    
    for(i=0, j=1; j<n_elem; i+=2, j+=2)
      {
      const eT tmp_i = A[i];
      const eT tmp_j = A[j];
      
      if(tmp_i > max_val) { max_val = tmp_i; }
      if(tmp_j > max_val) { max_val = tmp_j; }
      }
    
    if(i < n_elem)
      {
      const eT tmp_i = A[i];
      
      if(tmp_i > max_val) { max_val = tmp_i; }
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
        const eT tmp_i = P.at(0,i);
        const eT tmp_j = P.at(0,j);
        
        if(tmp_i > max_val) { max_val = tmp_i; }
        if(tmp_j > max_val) { max_val = tmp_j; }
        }
      
      if(i < n_cols)
        {
        const eT tmp_i = P.at(0,i);
        
        if(tmp_i > max_val) { max_val = tmp_i; }
        }
      }
    else
      {
      for(uword col=0; col < n_cols; ++col)
        {
        uword i,j;
        for(i=0, j=1; j < n_rows; i+=2, j+=2)
          {
          const eT tmp_i = P.at(i,col);
          const eT tmp_j = P.at(j,col);
          
          if(tmp_i > max_val) { max_val = tmp_i; }
          if(tmp_j > max_val) { max_val = tmp_j; }
          }
          
        if(i < n_rows)
          {
          const eT tmp_i = P.at(i,col);
          
          if(tmp_i > max_val) { max_val = tmp_i; }
          }
        }
      }
    }
  
  return max_val;
  }



template<typename T1>
inline
typename arma_not_cx<typename T1::elem_type>::result
op_max::max_with_index(const Proxy<T1>& P, uword& index_of_max_val)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_elem = P.get_n_elem();
  
  arma_debug_check( (n_elem == 0), "max(): given object has no elements" );
  
  eT    best_val   = priv::most_neg<eT>();
  uword best_index = 0;
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typedef typename Proxy<T1>::ea_type ea_type;
    
    ea_type A = P.get_ea();
    
    for(uword i=0; i<n_elem; ++i)
      {
      const eT tmp = A[i];
      
      if(tmp > best_val)  { best_val = tmp;  best_index = i; }
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    if(n_rows == 1)
      {
      for(uword i=0; i < n_cols; ++i)
        {
        const eT tmp = P.at(0,i);
        
        if(tmp > best_val)  { best_val = tmp;  best_index = i; }
        }
      }
    else
    if(n_cols == 1)
      {
      for(uword i=0; i < n_rows; ++i)
        {
        const eT tmp = P.at(i,0);
        
        if(tmp > best_val)  { best_val = tmp;  best_index = i; }
        }
      }
    else
      {
      uword count = 0;
      
      for(uword col=0; col < n_cols; ++col)
      for(uword row=0; row < n_rows; ++row)
        {
        const eT tmp = P.at(row,col);
        
        if(tmp > best_val)  { best_val = tmp;  best_index = count; }
        
        ++count;
        }
      }
    }
  
  index_of_max_val = best_index;
  
  return best_val;
  }



template<typename T>
inline
std::complex<T>
op_max::direct_max(const std::complex<T>* const X, const uword n_elem)
  {
  arma_extra_debug_sigprint();
  
  uword index   = 0;
  T   max_val = priv::most_neg<T>();
  
  for(uword i=0; i<n_elem; ++i)
    {
    const T tmp_val = std::abs(X[i]);
    
    if(tmp_val > max_val)
      {
      max_val = tmp_val;
      index   = i;
      }
    }
  
  return X[index];
  }



template<typename T>
inline
std::complex<T>
op_max::direct_max(const std::complex<T>* const X, const uword n_elem, uword& index_of_max_val)
  {
  arma_extra_debug_sigprint();
  
  uword index   = 0;
  T   max_val = priv::most_neg<T>();
  
  for(uword i=0; i<n_elem; ++i)
    {
    const T tmp_val = std::abs(X[i]);
    
    if(tmp_val > max_val)
      {
      max_val = tmp_val;
      index   = i;
      }
    }
  
  index_of_max_val = index;
  
  return X[index];
  }



template<typename T>
inline 
std::complex<T>
op_max::direct_max(const Mat< std::complex<T> >& X, const uword row)
  {
  arma_extra_debug_sigprint();
  
  const uword X_n_cols = X.n_cols;
  
  uword index   = 0;
  T   max_val = priv::most_neg<T>();
  
  for(uword col=0; col<X_n_cols; ++col)
    {
    const T tmp_val = std::abs(X.at(row,col));
    
    if(tmp_val > max_val)
      {
      max_val = tmp_val;
      index   = col;
      }
    }
  
  return X.at(row,index);
  }



template<typename T>
inline
std::complex<T>
op_max::max(const subview< std::complex<T> >& X)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (X.n_elem == 0), "max(): given object has no elements" );
  
  const Mat< std::complex<T> >& A = X.m;
  
  const uword X_n_rows = X.n_rows;
  const uword X_n_cols = X.n_cols;
  
  const uword start_row = X.aux_row1;
  const uword start_col = X.aux_col1;
  
  const uword end_row_p1 = start_row + X_n_rows;
  const uword end_col_p1 = start_col + X_n_cols;
  
  T max_val = priv::most_neg<T>();
  
  uword best_row = 0;
  uword best_col = 0;
    
  if(X_n_rows == 1)
    {
    best_col = 0;
    
    for(uword col=start_col; col < end_col_p1; ++col)
      {
      const T tmp_val = std::abs( A.at(start_row, col) );
      
      if(tmp_val > max_val)
        {
        max_val  = tmp_val;
        best_col = col;
        }
      }
    
    best_row = start_row;
    }
  else
    {
    for(uword col=start_col; col < end_col_p1; ++col)
    for(uword row=start_row; row < end_row_p1; ++row)
      {
      const T tmp_val = std::abs( A.at(row, col) );
      
      if(tmp_val > max_val)
        {
        max_val  = tmp_val;
        best_row = row;
        best_col = col;
        }
      }
    }
  
  return A.at(best_row, best_col);
  }



template<typename T1>
inline
typename arma_cx_only<typename T1::elem_type>::result
op_max::max(const Base<typename T1::elem_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type            eT;
  typedef typename get_pod_type<eT>::result T;
  
  const Proxy<T1> P(X.get_ref());
  
  const uword n_elem = P.get_n_elem();
  
  arma_debug_check( (n_elem == 0), "max(): given object has no elements" );
  
  T max_val = priv::most_neg<T>();
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typedef typename Proxy<T1>::ea_type ea_type;
    
    ea_type A = P.get_ea();
    
    uword index = 0;
    
    for(uword i=0; i<n_elem; ++i)
      {
      const T tmp = std::abs(A[i]);
      
      if(tmp > max_val)
        {
        max_val = tmp;
        index   = i;
        }
      }
    
    return( A[index] );
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    uword best_row = 0;
    uword best_col = 0;
    
    if(n_rows == 1)
      {
      for(uword col=0; col < n_cols; ++col)
        {
        const T tmp = std::abs(P.at(0,col));
        
        if(tmp > max_val)
          {
          max_val  = tmp;
          best_col = col;
          }
        }
      }
    else
      {
      for(uword col=0; col < n_cols; ++col)
      for(uword row=0; row < n_rows; ++row)
        {
        const T tmp = std::abs(P.at(row,col));
        
        if(tmp > max_val)
          {
          max_val = tmp;
          
          best_row = row;
          best_col = col;
          }
        }
      }
    
    return P.at(best_row, best_col);
    }
  }



template<typename T1>
inline
typename arma_cx_only<typename T1::elem_type>::result
op_max::max_with_index(const Proxy<T1>& P, uword& index_of_max_val)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type            eT;
  typedef typename get_pod_type<eT>::result T;
  
  const uword n_elem = P.get_n_elem();
  
  arma_debug_check( (n_elem == 0), "max(): given object has no elements" );
  
  T best_val = priv::most_neg<T>();
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typedef typename Proxy<T1>::ea_type ea_type;
    
    ea_type A = P.get_ea();
    
    uword best_index = 0;
    
    for(uword i=0; i<n_elem; ++i)
      {
      const T tmp = std::abs(A[i]);
      
      if(tmp > best_val)  { best_val = tmp;  best_index = i; }
      }
    
    index_of_max_val = best_index;
    
    return( A[best_index] );
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    uword best_row   = 0;
    uword best_col   = 0;
    uword best_index = 0;
    
    if(n_rows == 1)
      {
      for(uword col=0; col < n_cols; ++col)
        {
        const T tmp = std::abs(P.at(0,col));
        
        if(tmp > best_val)  { best_val = tmp;  best_col = col; }
        }
      
      best_index = best_col;
      }
    else
    if(n_cols == 1)
      {
      for(uword row=0; row < n_rows; ++row)
        {
        const T tmp = std::abs(P.at(row,0));
        
        if(tmp > best_val)  { best_val = tmp;  best_row = row; }
        }
      
      best_index = best_row;
      }
    else
      {
      uword count = 0;
      
      for(uword col=0; col < n_cols; ++col)
      for(uword row=0; row < n_rows; ++row)
        {
        const T tmp = std::abs(P.at(row,col));
        
        if(tmp > best_val)
          {
          best_val = tmp;
          
          best_row = row;
          best_col = col;
          
          best_index = count;
          }
        
        ++count;
        }
      }
    
    index_of_max_val = best_index;
    
    return P.at(best_row, best_col);
    }
  }



//! @}
