// Copyright (C) 2008-2015 Conrad Sanderson
// Copyright (C) 2008-2015 NICTA (www.nicta.com.au)
// Copyright (C) 2011 James Sanders
// Copyright (C) 2013 Ryan Curtin
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup subview
//! @{


template<typename eT>
inline
subview<eT>::~subview()
  {
  arma_extra_debug_sigprint();
  }


template<typename eT>
inline
subview<eT>::subview(const Mat<eT>& in_m, const uword in_row1, const uword in_col1, const uword in_n_rows, const uword in_n_cols)
  : m(in_m)
  , aux_row1(in_row1)
  , aux_col1(in_col1)
  , n_rows(in_n_rows)
  , n_cols(in_n_cols)
  , n_elem(in_n_rows*in_n_cols)
  {
  arma_extra_debug_sigprint();
  }



template<typename eT>
inline
void
subview<eT>::operator= (const eT val)
  {
  arma_extra_debug_sigprint();
  
  if(n_elem != 1)
    {
    arma_debug_assert_same_size(n_rows, n_cols, 1, 1, "copy into submatrix");
    }
  
  Mat<eT>& X = const_cast< Mat<eT>& >(m);
  
  X.at(aux_row1, aux_col1) = val;
  }



template<typename eT>
inline
void
subview<eT>::operator+= (const eT val)
  {
  arma_extra_debug_sigprint();
  
  const uword local_n_cols = n_cols;
  const uword local_n_rows = n_rows;
  
  if(local_n_rows == 1)
    {
    Mat<eT>& X = const_cast< Mat<eT>& >(m);
    
    const uword urow          = aux_row1;
    const uword start_col     = aux_col1;
    const uword end_col_plus1 = start_col + local_n_cols;
    
    uword ii,jj;
    for(ii=start_col, jj=start_col+1; jj < end_col_plus1; ii+=2, jj+=2)
      {
      X.at(urow, ii) += val;
      X.at(urow, jj) += val;
      }
    
    if(ii < end_col_plus1)
      {
      X.at(urow, ii) += val;
      }
    }
  else
    {
    for(uword ucol=0; ucol < local_n_cols; ++ucol)
      {
      arrayops::inplace_plus( colptr(ucol), val, local_n_rows );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::operator-= (const eT val)
  {
  arma_extra_debug_sigprint();
  
  const uword local_n_cols = n_cols;
  const uword local_n_rows = n_rows;
  
  if(local_n_rows == 1)
    {
    Mat<eT>& X = const_cast< Mat<eT>& >(m);
    
    const uword urow          = aux_row1;
    const uword start_col     = aux_col1;
    const uword end_col_plus1 = start_col + local_n_cols;
    
    uword ii,jj;
    for(ii=start_col, jj=start_col+1; jj < end_col_plus1; ii+=2, jj+=2)
      {
      X.at(urow, ii) -= val;
      X.at(urow, jj) -= val;
      }
    
    if(ii < end_col_plus1)
      {
      X.at(urow, ii) -= val;
      }
    }
  else
    {
    for(uword ucol=0; ucol < local_n_cols; ++ucol)
      {
      arrayops::inplace_minus( colptr(ucol), val, local_n_rows );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::operator*= (const eT val)
  {
  arma_extra_debug_sigprint();
  
  const uword local_n_cols = n_cols;
  const uword local_n_rows = n_rows;
  
  if(local_n_rows == 1)
    {
    Mat<eT>& X = const_cast< Mat<eT>& >(m);
    
    const uword urow          = aux_row1;
    const uword start_col     = aux_col1;
    const uword end_col_plus1 = start_col + local_n_cols;
    
    uword ii,jj;
    for(ii=start_col, jj=start_col+1; jj < end_col_plus1; ii+=2, jj+=2)
      {
      X.at(urow, ii) *= val;
      X.at(urow, jj) *= val;
      }
    
    if(ii < end_col_plus1)
      {
      X.at(urow, ii) *= val;
      }
    }
  else
    {
    for(uword ucol=0; ucol < local_n_cols; ++ucol)
      {
      arrayops::inplace_mul( colptr(ucol), val, local_n_rows );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::operator/= (const eT val)
  {
  arma_extra_debug_sigprint();
  
  const uword local_n_cols = n_cols;
  const uword local_n_rows = n_rows;
  
  if(local_n_rows == 1)
    {
    Mat<eT>& X = const_cast< Mat<eT>& >(m);
    
    const uword urow          = aux_row1;
    const uword start_col     = aux_col1;
    const uword end_col_plus1 = start_col + local_n_cols;
    
    uword ii,jj;
    for(ii=start_col, jj=start_col+1; jj < end_col_plus1; ii+=2, jj+=2)
      {
      X.at(urow, ii) /= val;
      X.at(urow, jj) /= val;
      }
    
    if(ii < end_col_plus1)
      {
      X.at(urow, ii) /= val;
      }
    }
  else
    {
    for(uword ucol=0; ucol < local_n_cols; ++ucol)
      {
      arrayops::inplace_div( colptr(ucol), val, local_n_rows );
      }
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  const Proxy<T1> P(in.get_ref());
  
  subview<eT>& s = *this;
  
  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;
    
  arma_debug_assert_same_size(s, P, "copy into submatrix");
  
  const bool is_alias = P.is_alias(s.m);
  
  arma_extra_debug_warn(is_alias, "aliasing detected");
  
  if( (is_Mat<typename Proxy<T1>::stored_type>::value == true) || (is_alias == true) )
    {
    const unwrap_check<typename Proxy<T1>::stored_type> tmp(P.Q, is_alias);
    const Mat<eT>& x = tmp.M;
    
    if(s_n_rows == 1)
      {
      const eT* x_mem = x.memptr();
      
      Mat<eT>& A = const_cast< Mat<eT>& >(m);
      
      const uword urow      = aux_row1;
      const uword start_col = aux_col1;
      
      uword ii,jj;
      for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
        {
        A.at(urow, start_col+ii) = x_mem[ii];
        A.at(urow, start_col+jj) = x_mem[jj];
        }
      
      if(ii < s_n_cols)
        {
        A.at(urow, start_col+ii) = x_mem[ii];
        }
      }
    else
      {
      for(uword ucol=0; ucol < s_n_cols; ++ucol)
        {
        arrayops::copy( s.colptr(ucol), x.colptr(ucol), s_n_rows );
        }
      }
    }
  else
    {
    if(s_n_rows == 1)
      {
      Mat<eT>& A = const_cast< Mat<eT>& >(m);
      
      const uword urow      = aux_row1;
      const uword start_col = aux_col1;
      
      uword ii,jj;
      for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
        {
        const eT tmp1 = (Proxy<T1>::prefer_at_accessor) ? P.at(0,ii) : P[ii];
        const eT tmp2 = (Proxy<T1>::prefer_at_accessor) ? P.at(0,jj) : P[jj];
        
        A.at(urow, start_col+ii) = tmp1;
        A.at(urow, start_col+jj) = tmp2;
        }
      
      if(ii < s_n_cols)
        {
        A.at(urow, start_col+ii) = (Proxy<T1>::prefer_at_accessor) ? P.at(0,ii) : P[ii];
        }
      }
    else
      {
      for(uword ucol=0; ucol < s_n_cols; ++ucol)
        {
        eT* s_col_data = s.colptr(ucol);
        
        uword ii,jj;
        for(ii=0, jj=1; jj < s_n_rows; ii+=2, jj+=2)
          {
          const eT tmp1 = P.at(ii,ucol);
          const eT tmp2 = P.at(jj,ucol);
          
          s_col_data[ii] = tmp1;
          s_col_data[jj] = tmp2;
          }
        
        if(ii < s_n_rows)
          {
          s_col_data[ii] = P.at(ii,ucol);
          }
        }
      }
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator+= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  const Proxy<T1> P(in.get_ref());
  
  subview<eT>& s = *this;
  
  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;
  
  arma_debug_assert_same_size(s, P, "addition");
  
  const bool is_alias = P.is_alias(s.m);
  
  arma_extra_debug_warn(is_alias, "aliasing detected");
  
  if( (is_Mat<typename Proxy<T1>::stored_type>::value == true) || (is_alias == true) )
    {
    const unwrap_check<typename Proxy<T1>::stored_type> tmp(P.Q, is_alias);
    const Mat<eT>& x = tmp.M;
    
    if(s_n_rows == 1)
      {
      const eT* x_mem = x.memptr();
      
      Mat<eT>& A = const_cast< Mat<eT>& >(m);
      
      const uword urow      = aux_row1;
      const uword start_col = aux_col1;
      
      uword ii,jj;
      for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
        {
        A.at(urow, start_col+ii) += x_mem[ii];
        A.at(urow, start_col+jj) += x_mem[jj];
        }
      
      if(ii < s_n_cols)
        {
        A.at(urow, start_col+ii) += x_mem[ii];
        }
      }
    else
      {
      for(uword ucol=0; ucol < s_n_cols; ++ucol)
        {
        arrayops::inplace_plus( s.colptr(ucol), x.colptr(ucol), s_n_rows );
        }
      }
    }
  else
    {
    if(s_n_rows == 1)
      {
      Mat<eT>& A = const_cast< Mat<eT>& >(m);
      
      const uword urow      = aux_row1;
      const uword start_col = aux_col1;
      
      uword ii,jj;
      for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
        {
        const eT tmp1 = (Proxy<T1>::prefer_at_accessor) ? P.at(0,ii) : P[ii];
        const eT tmp2 = (Proxy<T1>::prefer_at_accessor) ? P.at(0,jj) : P[jj];
        
        A.at(urow, start_col+ii) += tmp1;
        A.at(urow, start_col+jj) += tmp2;
        }
      
      if(ii < s_n_cols)
        {
        A.at(urow, start_col+ii) += (Proxy<T1>::prefer_at_accessor) ? P.at(0,ii) : P[ii];
        }
      }
    else
      {
      for(uword ucol=0; ucol < s_n_cols; ++ucol)
        {
        eT* s_col_data = s.colptr(ucol);
        
        uword ii,jj;
        for(ii=0, jj=1; jj < s_n_rows; ii+=2, jj+=2)
          {
          const eT val1 = P.at(ii,ucol);
          const eT val2 = P.at(jj,ucol);
          
          s_col_data[ii] += val1;
          s_col_data[jj] += val2;
          }
        
        if(ii < s_n_rows)
          {
          s_col_data[ii] += P.at(ii,ucol);
          }
        }
      }
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator-= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  const Proxy<T1> P(in.get_ref());
  
  subview<eT>& s = *this;
  
  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;
  
  arma_debug_assert_same_size(s, P, "subtraction");
  
  const bool is_alias = P.is_alias(s.m);
  
  arma_extra_debug_warn(is_alias, "aliasing detected");
  
  if( (is_Mat<typename Proxy<T1>::stored_type>::value == true) || (is_alias == true) )
    {
    const unwrap_check<typename Proxy<T1>::stored_type> tmp(P.Q, is_alias);
    const Mat<eT>& x = tmp.M;
    
    if(s_n_rows == 1)
      {
      const eT* x_mem = x.memptr();
      
      Mat<eT>& A = const_cast< Mat<eT>& >(m);
      
      const uword urow      = aux_row1;
      const uword start_col = aux_col1;
      
      uword ii,jj;
      for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
        {
        A.at(urow, start_col+ii) -= x_mem[ii];
        A.at(urow, start_col+jj) -= x_mem[jj];
        }
      
      if(ii < s_n_cols)
        {
        A.at(urow, start_col+ii) -= x_mem[ii];
        }
      }
    else
      {
      for(uword ucol=0; ucol < s_n_cols; ++ucol)
        {
        arrayops::inplace_minus( s.colptr(ucol), x.colptr(ucol), s_n_rows );
        }
      }
    }
  else
    {
    if(s_n_rows == 1)
      {
      Mat<eT>& A = const_cast< Mat<eT>& >(m);
      
      const uword urow      = aux_row1;
      const uword start_col = aux_col1;
      
      uword ii,jj;
      for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
        {
        const eT tmp1 = (Proxy<T1>::prefer_at_accessor) ? P.at(0,ii) : P[ii];
        const eT tmp2 = (Proxy<T1>::prefer_at_accessor) ? P.at(0,jj) : P[jj];
        
        A.at(urow, start_col+ii) -= tmp1;
        A.at(urow, start_col+jj) -= tmp2;
        }
      
      if(ii < s_n_cols)
        {
        A.at(urow, start_col+ii) -= (Proxy<T1>::prefer_at_accessor) ? P.at(0,ii) : P[ii];
        }
      }
    else
      {
      for(uword ucol=0; ucol < s_n_cols; ++ucol)
        {
        eT* s_col_data = s.colptr(ucol);
        
        uword ii,jj;
        for(ii=0, jj=1; jj < s_n_rows; ii+=2, jj+=2)
          {
          const eT val1 = P.at(ii,ucol);
          const eT val2 = P.at(jj,ucol);
          
          s_col_data[ii] -= val1;
          s_col_data[jj] -= val2;
          }
        
        if(ii < s_n_rows)
          {
          s_col_data[ii] -= P.at(ii,ucol);
          }
        }
      }
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator%= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  const Proxy<T1> P(in.get_ref());
  
  subview<eT>& s = *this;
  
  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;
  
  arma_debug_assert_same_size(s, P, "element-wise multiplication");
  
  const bool is_alias = P.is_alias(s.m);
  
  arma_extra_debug_warn(is_alias, "aliasing detected");
  
  if( (is_Mat<typename Proxy<T1>::stored_type>::value == true) || (is_alias == true) )
    {
    const unwrap_check<typename Proxy<T1>::stored_type> tmp(P.Q, is_alias);
    const Mat<eT>& x = tmp.M;
    
    if(s_n_rows == 1)
      {
      const eT* x_mem = x.memptr();
      
      Mat<eT>& A = const_cast< Mat<eT>& >(m);
      
      const uword urow      = aux_row1;
      const uword start_col = aux_col1;
      
      uword ii,jj;
      for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
        {
        A.at(urow, start_col+ii) *= x_mem[ii];
        A.at(urow, start_col+jj) *= x_mem[jj];
        }
      
      if(ii < s_n_cols)
        {
        A.at(urow, start_col+ii) *= x_mem[ii];
        }
      }
    else
      {
      for(uword ucol=0; ucol < s_n_cols; ++ucol)
        {
        arrayops::inplace_mul( s.colptr(ucol), x.colptr(ucol), s_n_rows );
        }
      }
    }
  else
    {
    if(s_n_rows == 1)
      {
      Mat<eT>& A = const_cast< Mat<eT>& >(m);
      
      const uword urow      = aux_row1;
      const uword start_col = aux_col1;
      
      uword ii,jj;
      for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
        {
        const eT tmp1 = (Proxy<T1>::prefer_at_accessor) ? P.at(0,ii) : P[ii];
        const eT tmp2 = (Proxy<T1>::prefer_at_accessor) ? P.at(0,jj) : P[jj];
        
        A.at(urow, start_col+ii) *= tmp1;
        A.at(urow, start_col+jj) *= tmp2;
        }
      
      if(ii < s_n_cols)
        {
        A.at(urow, start_col+ii) *= (Proxy<T1>::prefer_at_accessor) ? P.at(0,ii) : P[ii];
        }
      }
    else
      {
      for(uword ucol=0; ucol < s_n_cols; ++ucol)
        {
        eT* s_col_data = s.colptr(ucol);
        
        uword ii,jj;
        for(ii=0, jj=1; jj < s_n_rows; ii+=2, jj+=2)
          {
          const eT val1 = P.at(ii,ucol);
          const eT val2 = P.at(jj,ucol);
          
          s_col_data[ii] *= val1;
          s_col_data[jj] *= val2;
          }
        
        if(ii < s_n_rows)
          {
          s_col_data[ii] *= P.at(ii,ucol);
          }
        }
      }
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator/= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  const Proxy<T1> P(in.get_ref());
  
  subview<eT>& s = *this;
  
  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;
  
  arma_debug_assert_same_size(s, P, "element-wise division");
  
  const bool is_alias = P.is_alias(s.m);
  
  arma_extra_debug_warn(is_alias, "aliasing detected");
  
  if( (is_Mat<typename Proxy<T1>::stored_type>::value == true) || (is_alias == true) )
    {
    const unwrap_check<typename Proxy<T1>::stored_type> tmp(P.Q, is_alias);
    const Mat<eT>& x = tmp.M;
    
    if(s_n_rows == 1)
      {
      const eT* x_mem = x.memptr();
      
      Mat<eT>& A = const_cast< Mat<eT>& >(m);
      
      const uword urow      = aux_row1;
      const uword start_col = aux_col1;
      
      uword ii,jj;
      for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
        {
        A.at(urow, start_col+ii) /= x_mem[ii];
        A.at(urow, start_col+jj) /= x_mem[jj];
        }
      
      if(ii < s_n_cols)
        {
        A.at(urow, start_col+ii) /= x_mem[ii];
        }
      }
    else
      {
      for(uword ucol=0; ucol < s_n_cols; ++ucol)
        {
        arrayops::inplace_div( s.colptr(ucol), x.colptr(ucol), s_n_rows );
        }
      }
    }
  else
    {
    if(s_n_rows == 1)
      {
      Mat<eT>& A = const_cast< Mat<eT>& >(m);
      
      const uword urow      = aux_row1;
      const uword start_col = aux_col1;
      
      uword ii,jj;
      for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
        {
        const eT tmp1 = (Proxy<T1>::prefer_at_accessor) ? P.at(0,ii) : P[ii];
        const eT tmp2 = (Proxy<T1>::prefer_at_accessor) ? P.at(0,jj) : P[jj];
        
        A.at(urow, start_col+ii) /= tmp1;
        A.at(urow, start_col+jj) /= tmp2;
        }
      
      if(ii < s_n_cols)
        {
        A.at(urow, start_col+ii) /= (Proxy<T1>::prefer_at_accessor) ? P.at(0,ii) : P[ii];
        }
      }
    else
      {
      for(uword ucol=0; ucol < s_n_cols; ++ucol)
        {
        eT* s_col_data = s.colptr(ucol);
        
        uword ii,jj;
        for(ii=0, jj=1; jj < s_n_rows; ii+=2, jj+=2)
          {
          const eT val1 = P.at(ii,ucol);
          const eT val2 = P.at(jj,ucol);
          
          s_col_data[ii] /= val1;
          s_col_data[jj] /= val2;
          }
        
        if(ii < s_n_rows)
          {
          s_col_data[ii] /= P.at(ii,ucol);
          }
        }
      }
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator=(const SpBase<eT, T1>& x)
  {
  arma_extra_debug_sigprint();
  
  const SpProxy<T1> p(x.get_ref());
  
  arma_debug_assert_same_size(n_rows, n_cols, p.get_n_rows(), p.get_n_cols(), "copy into submatrix");
  
  // Clear the subview.
  zeros();
  
  // Iterate through the sparse subview and set the nonzero values appropriately.
  typename SpProxy<T1>::const_iterator_type cit = p.begin();
  
  while (cit != p.end())
    {
    at(cit.row(), cit.col()) = *cit;
    ++cit;
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator+=(const SpBase<eT, T1>& x)
  {
  arma_extra_debug_sigprint();
  
  const SpProxy<T1> p(x.get_ref());
  
  arma_debug_assert_same_size(n_rows, n_cols, p.get_n_rows(), p.get_n_cols(), "addition");
  
  // Iterate through the sparse subview and add its values.
  typename SpProxy<T1>::const_iterator_type cit = p.begin();
  
  while (cit != p.end())
    {
    at(cit.row(), cit.col()) += *cit;
    ++cit;
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator-=(const SpBase<eT, T1>& x)
  {
  arma_extra_debug_sigprint();
  
  const SpProxy<T1> p(x.get_ref());
  
  arma_debug_assert_same_size(n_rows, n_cols, p.get_n_rows(), p.get_n_cols(), "subtraction");
  
  // Iterate through the sparse subview and subtract its values.
  typename SpProxy<T1>::const_iterator_type cit = p.begin();
  
  while (cit != p.end())
    {
    at(cit.row(), cit.col()) -= *cit;
    ++cit;
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator%=(const SpBase<eT, T1>& x)
  {
  arma_extra_debug_sigprint();
  
  // Temporary sparse matrix to hold the values we need.
  SpMat<eT> tmp = x.get_ref();
  
  arma_debug_assert_same_size(n_rows, n_cols, tmp.n_rows, tmp.n_cols, "element-wise multiplication");
  
  // Iterate over nonzero values.
  // Any zero values in the sparse expression will result in a zero in our subview.
  typename SpMat<eT>::const_iterator cit = tmp.begin();
  
  while (cit != tmp.end())
    {
    // Set elements before this one to zero.
    tmp.at(cit.row(), cit.col()) *= at(cit.row(), cit.col());
    ++cit;
    }
  
  // Now set the subview equal to that.
  *this = tmp;
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator/=(const SpBase<eT, T1>& x)
  {
  arma_extra_debug_sigprint();
  
  const SpProxy<T1> p(x.get_ref());
  
  arma_debug_assert_same_size(n_rows, n_cols, p.get_n_rows(), p.get_n_cols(), "element-wise division");
  
  // This is probably going to fill your subview with a bunch of NaNs,
  // so I'm not going to bother to implement it fast.
  // You can have slow NaNs.  They're fine too.
  for (uword c = 0; c < n_cols; ++c)
  for (uword r = 0; r < n_rows; ++r)
    {
    at(r, c) /= p.at(r, c);
    }
  }



//! x.submat(...) = y.submat(...)
template<typename eT>
inline
void
subview<eT>::operator= (const subview<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  if(check_overlap(x))
    {
    const Mat<eT> tmp(x);
    
    (*this).operator=(tmp);
    
    return;
    }
  
  subview<eT>& s = *this;
  
  arma_debug_assert_same_size(s, x, "copy into submatrix");
  
  const uword s_n_cols = s.n_cols;
  const uword s_n_rows = s.n_rows;
  
  if(s_n_rows == 1)
    {
          Mat<eT>& A = const_cast< Mat<eT>& >(s.m);
    const Mat<eT>& B = x.m;
    
    const uword row_A = s.aux_row1;
    const uword row_B = x.aux_row1;
    
    const uword start_col_A = s.aux_col1;
    const uword start_col_B = x.aux_col1;
    
    uword ii,jj;
    for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
      {
      const eT tmp1 = B.at(row_B, start_col_B + ii);
      const eT tmp2 = B.at(row_B, start_col_B + jj);
      
      A.at(row_A, start_col_A + ii) = tmp1;
      A.at(row_A, start_col_A + jj) = tmp2;
      }
    
    if(ii < s_n_cols)
      {
      A.at(row_A, start_col_A + ii) = B.at(row_B, start_col_B + ii);
      }
    }
  else
    {
    for(uword ucol=0; ucol < s_n_cols; ++ucol)
      {
      arrayops::copy( s.colptr(ucol), x.colptr(ucol), s_n_rows );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::operator+= (const subview<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  if(check_overlap(x))
    {
    const Mat<eT> tmp(x);
    
    (*this).operator+=(tmp);
    
    return;
    }
  
  subview<eT>& s = *this;
  
  arma_debug_assert_same_size(s, x, "addition");
  
  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;
  
  if(s_n_rows == 1)
    {
          Mat<eT>& A = const_cast< Mat<eT>& >(s.m);
    const Mat<eT>& B = x.m;
    
    const uword row_A = s.aux_row1;
    const uword row_B = x.aux_row1;
    
    const uword start_col_A = s.aux_col1;
    const uword start_col_B = x.aux_col1;
    
    uword ii,jj;
    
    for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
      {
      const eT tmp1 = B.at(row_B, start_col_B + ii);
      const eT tmp2 = B.at(row_B, start_col_B + jj);
      
      A.at(row_A, start_col_A + ii) += tmp1;
      A.at(row_A, start_col_A + jj) += tmp2;
      }
    
    if(ii < s_n_cols)
      {
      A.at(row_A, start_col_A + ii) += B.at(row_B, start_col_B + ii);
      }
    }
  else
    {
    for(uword ucol=0; ucol < s_n_cols; ++ucol)
      {
      arrayops::inplace_plus( s.colptr(ucol), x.colptr(ucol), s_n_rows );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::operator-= (const subview<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  if(check_overlap(x))
    {
    const Mat<eT> tmp(x);
    
    (*this).operator-=(tmp);
    
    return;
    }
  
  subview<eT>& s = *this;
  
  arma_debug_assert_same_size(s, x, "subtraction");
  
  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;
  
  if(s_n_rows == 1)
    {
          Mat<eT>& A = const_cast< Mat<eT>& >(s.m);
    const Mat<eT>& B = x.m;
    
    const uword row_A = s.aux_row1;
    const uword row_B = x.aux_row1;
    
    const uword start_col_A = s.aux_col1;
    const uword start_col_B = x.aux_col1;
    
    uword ii,jj;
    for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
      {
      const eT tmp1 = B.at(row_B, start_col_B + ii);
      const eT tmp2 = B.at(row_B, start_col_B + jj);
      
      A.at(row_A, start_col_A + ii) -= tmp1;
      A.at(row_A, start_col_A + jj) -= tmp2;
      }
    
    if(ii < s_n_cols)
      {
      A.at(row_A, start_col_A + ii) -= B.at(row_B, start_col_B + ii);
      }
    }
  else
    {
    for(uword ucol=0; ucol < s_n_cols; ++ucol)
      {
      arrayops::inplace_minus( s.colptr(ucol), x.colptr(ucol), s_n_rows );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::operator%= (const subview& x)
  {
  arma_extra_debug_sigprint();
  
  if(check_overlap(x))
    {
    const Mat<eT> tmp(x);
    
    (*this).operator%=(tmp);
    
    return;
    }
  
  subview<eT>& s = *this;
  
  arma_debug_assert_same_size(s, x, "element-wise multiplication");
  
  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;
  
  if(s_n_rows == 1)
    {
          Mat<eT>& A = const_cast< Mat<eT>& >(s.m);
    const Mat<eT>& B = x.m;
    
    const uword row_A = s.aux_row1;
    const uword row_B = x.aux_row1;
    
    const uword start_col_A = s.aux_col1;
    const uword start_col_B = x.aux_col1;
    
    uword ii,jj;
    for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
      {
      const eT tmp1 = B.at(row_B, start_col_B + ii);
      const eT tmp2 = B.at(row_B, start_col_B + jj);
      
      A.at(row_A, start_col_A + ii) *= tmp1;
      A.at(row_A, start_col_A + jj) *= tmp2;
      }
    
    if(ii < s_n_cols)
      {
      A.at(row_A, start_col_A + ii) *= B.at(row_B, start_col_B + ii);
      }
    }
  else
    {
    for(uword ucol=0; ucol < s_n_cols; ++ucol)
      {
      arrayops::inplace_mul( s.colptr(ucol), x.colptr(ucol), s_n_rows );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::operator/= (const subview& x)
  {
  arma_extra_debug_sigprint();
  
  if(check_overlap(x))
    {
    const Mat<eT> tmp(x);
    
    (*this).operator/=(tmp);
    
    return;
    }
  
  subview<eT>& s = *this;
  
  arma_debug_assert_same_size(s, x, "element-wise division");
  
  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;
  
  if(s_n_rows == 1)
    {
          Mat<eT>& A = const_cast< Mat<eT>& >(s.m);
    const Mat<eT>& B = x.m;
    
    const uword row_A = s.aux_row1;
    const uword row_B = x.aux_row1;
    
    const uword start_col_A = s.aux_col1;
    const uword start_col_B = x.aux_col1;
    
    uword ii,jj;
    for(ii=0, jj=1; jj < s_n_cols; ii+=2, jj+=2)
      {
      const eT tmp1 = B.at(row_B, start_col_B + ii);
      const eT tmp2 = B.at(row_B, start_col_B + jj);
      
      A.at(row_A, start_col_A + ii) /= tmp1;
      A.at(row_A, start_col_A + jj) /= tmp2;
      }
    
    if(ii < s_n_cols)
      {
      A.at(row_A, start_col_A + ii) /= B.at(row_B, start_col_B + ii);
      }
    }
  else
    {
    for(uword ucol=0; ucol < s_n_cols; ++ucol)
      {
      arrayops::inplace_div( s.colptr(ucol), x.colptr(ucol), s_n_rows );
      }
    }
  }



template<typename eT>
template<typename T1, typename gen_type>
inline
typename enable_if2< is_same_type<typename T1::elem_type, eT>::value, void>::result
subview<eT>::operator= (const Gen<T1,gen_type>& in)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(n_rows, n_cols, in.n_rows, in.n_cols, "copy into submatrix");
  
  in.apply(*this);
  }



//! transform each element in the subview using a functor
template<typename eT>
template<typename functor>
inline
void
subview<eT>::transform(functor F)
  {
  arma_extra_debug_sigprint();
  
  const uword local_n_cols = n_cols;
  const uword local_n_rows = n_rows;
  
  Mat<eT>& X = const_cast< Mat<eT>& >(m);
  
  if(local_n_rows == 1)
    {
    const uword urow          = aux_row1;
    const uword start_col     = aux_col1;
    const uword end_col_plus1 = start_col + local_n_cols;
    
    for(uword ucol = start_col; ucol < end_col_plus1; ++ucol)
      {
      X.at(urow, ucol) = eT( F( X.at(urow, ucol) ) );
      }
    }
  else
    {
    const uword start_col = aux_col1;
    const uword start_row = aux_row1;
    
    const uword end_col_plus1 = start_col + local_n_cols;
    const uword end_row_plus1 = start_row + local_n_rows;
    
    for(uword ucol = start_col; ucol < end_col_plus1; ++ucol)
    for(uword urow = start_row; urow < end_row_plus1; ++urow)
      {
      X.at(urow, ucol) = eT( F( X.at(urow, ucol) ) );
      }
    }
  }



//! imbue (fill) the subview with values provided by a functor
template<typename eT>
template<typename functor>
inline
void
subview<eT>::imbue(functor F)
  {
  arma_extra_debug_sigprint();
  
  const uword local_n_cols = n_cols;
  const uword local_n_rows = n_rows;
  
  Mat<eT>& X = const_cast< Mat<eT>& >(m);
  
  if(local_n_rows == 1)
    {
    const uword urow          = aux_row1;
    const uword start_col     = aux_col1;
    const uword end_col_plus1 = start_col + local_n_cols;
    
    for(uword ucol = start_col; ucol < end_col_plus1; ++ucol)
      {
      X.at(urow, ucol) = eT( F() );
      }
    }
  else
    {
    const uword start_col = aux_col1;
    const uword start_row = aux_row1;
    
    const uword end_col_plus1 = start_col + local_n_cols;
    const uword end_row_plus1 = start_row + local_n_rows;
    
    for(uword ucol = start_col; ucol < end_col_plus1; ++ucol)
    for(uword urow = start_row; urow < end_row_plus1; ++urow)
      {
      X.at(urow, ucol) = eT( F() );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::fill(const eT val)
  {
  arma_extra_debug_sigprint();
  
  const uword local_n_cols = n_cols;
  const uword local_n_rows = n_rows;
  
  if(local_n_rows == 1)
    {
    Mat<eT>& X = const_cast< Mat<eT>& >(m);
    
    const uword urow          = aux_row1;
    const uword start_col     = aux_col1;
    const uword end_col_plus1 = start_col + local_n_cols;
    
    uword ii,jj;
    for(ii=start_col, jj=start_col+1; jj < end_col_plus1; ii+=2, jj+=2)
      {
      X.at(urow, ii) = val;
      X.at(urow, jj) = val;
      }
    
    if(ii < end_col_plus1)
      {
      X.at(urow, ii) = val;
      }
    }
  else
    {
    for(uword ucol=0; ucol < local_n_cols; ++ucol)
      {
      arrayops::inplace_set( colptr(ucol), val, local_n_rows );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::zeros()
  {
  arma_extra_debug_sigprint();
  
  const uword local_n_cols = n_cols;
  const uword local_n_rows = n_rows;
  
  if(local_n_rows == 1)
    {
    (*this).fill(eT(0));
    }
  else
    {
    for(uword ucol=0; ucol < local_n_cols; ++ucol)
      {
      arrayops::fill_zeros( colptr(ucol), local_n_rows );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::ones()
  {
  arma_extra_debug_sigprint();
  
  (*this).fill(eT(1));
  }



template<typename eT>
inline
void
subview<eT>::eye()
  {
  arma_extra_debug_sigprint();
  
  (*this).zeros();
  
  const uword N = (std::min)(n_rows, n_cols);
  
  for(uword ii=0; ii < N; ++ii)
    {
    at(ii,ii) = eT(1);
    }
  }



template<typename eT>
inline
void
subview<eT>::randu()
  {
  arma_extra_debug_sigprint();
  
  const uword local_n_rows = n_rows;
  const uword local_n_cols = n_cols;
  
  if(local_n_rows == 1)
    {
    for(uword ii=0; ii < local_n_cols; ++ii)
      {
      at(0,ii) = eT(arma_rng::randu<eT>());
      }
    }
  else
    {
    for(uword ii=0; ii < local_n_cols; ++ii)
      {
      arma_rng::randu<eT>::fill( colptr(ii), local_n_rows );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::randn()
  {
  arma_extra_debug_sigprint();
  
  const uword local_n_rows = n_rows;
  const uword local_n_cols = n_cols;
  
  if(local_n_rows == 1)
    {
    for(uword ii=0; ii < local_n_cols; ++ii)
      {
      at(0,ii) = eT(arma_rng::randn<eT>());
      }
    }
  else
    {
    for(uword ii=0; ii < local_n_cols; ++ii)
      {
      arma_rng::randn<eT>::fill( colptr(ii), local_n_rows );
      }
    }
  }



template<typename eT>
inline
eT
subview<eT>::at_alt(const uword ii) const
  {
  return operator[](ii);
  }



template<typename eT>
inline
eT&
subview<eT>::operator[](const uword ii)
  {
  const uword in_col = ii / n_rows;
  const uword in_row = ii % n_rows;
  
  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;
  
  return access::rw( (const_cast< Mat<eT>& >(m)).mem[index] );
  }



template<typename eT>
inline
eT
subview<eT>::operator[](const uword ii) const
  {
  const uword in_col = ii / n_rows;
  const uword in_row = ii % n_rows;
  
  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;
  
  return m.mem[index];
  }



template<typename eT>
inline
eT&
subview<eT>::operator()(const uword ii)
  {
  arma_debug_check( (ii >= n_elem), "subview::operator(): index out of bounds");
    
  const uword in_col = ii / n_rows;
  const uword in_row = ii % n_rows;
  
  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;
  
  return access::rw( (const_cast< Mat<eT>& >(m)).mem[index] );
  }



template<typename eT>
inline
eT
subview<eT>::operator()(const uword ii) const
  {
  arma_debug_check( (ii >= n_elem), "subview::operator(): index out of bounds");
  
  const uword in_col = ii / n_rows;
  const uword in_row = ii % n_rows;
  
  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;
  
  return m.mem[index];
  }



template<typename eT>
inline
eT&
subview<eT>::operator()(const uword in_row, const uword in_col)
  {
  arma_debug_check( ((in_row >= n_rows) || (in_col >= n_cols)), "subview::operator(): index out of bounds");
  
  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;
  
  return access::rw( (const_cast< Mat<eT>& >(m)).mem[index] );
  }



template<typename eT>
inline
eT
subview<eT>::operator()(const uword in_row, const uword in_col) const
  {
  arma_debug_check( ((in_row >= n_rows) || (in_col >= n_cols)), "subview::operator(): index out of bounds");
  
  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;
  
  return m.mem[index];
  }



template<typename eT>
inline
eT&
subview<eT>::at(const uword in_row, const uword in_col)
  {
  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;
  
  return access::rw( (const_cast< Mat<eT>& >(m)).mem[index] );
  }



template<typename eT>
inline
eT
subview<eT>::at(const uword in_row, const uword in_col) const
  {
  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;
  
  return m.mem[index];
  }



template<typename eT>
arma_inline
eT*
subview<eT>::colptr(const uword in_col)
  {
  return & access::rw((const_cast< Mat<eT>& >(m)).mem[ (in_col + aux_col1)*m.n_rows + aux_row1 ]);
  }



template<typename eT>
arma_inline
const eT*
subview<eT>::colptr(const uword in_col) const
  {
  return & m.mem[ (in_col + aux_col1)*m.n_rows + aux_row1 ];
  }



template<typename eT>
inline
bool
subview<eT>::check_overlap(const subview<eT>& x) const
  {
  const subview<eT>& s = *this;
  
  if(&s.m != &x.m)
    {
    return false;
    }
  else
    {
    if( (s.n_elem == 0) || (x.n_elem == 0) )
      {
      return false;
      }
    else
      {
      const uword s_row_start  = s.aux_row1;
      const uword s_row_end_p1 = s_row_start + s.n_rows;
      
      const uword s_col_start  = s.aux_col1;
      const uword s_col_end_p1 = s_col_start + s.n_cols;
      
      
      const uword x_row_start  = x.aux_row1;
      const uword x_row_end_p1 = x_row_start + x.n_rows;
      
      const uword x_col_start  = x.aux_col1;
      const uword x_col_end_p1 = x_col_start + x.n_cols;
      
      
      const bool outside_rows = ( (x_row_start >= s_row_end_p1) || (s_row_start >= x_row_end_p1) );
      const bool outside_cols = ( (x_col_start >= s_col_end_p1) || (s_col_start >= x_col_end_p1) );
      
      return ( (outside_rows == false) && (outside_cols == false) );
      }
    }
  }



template<typename eT>
inline
arma_warn_unused
bool
subview<eT>::is_vec() const
  {
  return ( (n_rows == 1) || (n_cols == 1) );
  }



template<typename eT>
inline
arma_warn_unused
bool
subview<eT>::is_finite() const
  {
  arma_extra_debug_sigprint();
  
  const uword local_n_rows = n_rows;
  const uword local_n_cols = n_cols;
  
  for(uword ii=0; ii<local_n_cols; ++ii)
    {
    if(arrayops::is_finite(colptr(ii), local_n_rows) == false)  { return false; }
    }
  
  return true;
  }



//! X = Y.submat(...)
template<typename eT>
inline
void
subview<eT>::extract(Mat<eT>& out, const subview<eT>& in)
  {
  arma_extra_debug_sigprint();
  
  // NOTE: we're assuming that the matrix has already been set to the correct size and there is no aliasing;
  // size setting and alias checking is done by either the Mat contructor or operator=()
  
  const uword n_rows = in.n_rows;  // number of rows in the subview
  const uword n_cols = in.n_cols;  // number of columns in the subview
  
  arma_extra_debug_print(arma_boost::format("out.n_rows = %d   out.n_cols = %d    in.m.n_rows = %d  in.m.n_cols = %d") % out.n_rows % out.n_cols % in.m.n_rows % in.m.n_cols );
  
  
  if(in.is_vec() == true)
    {
    if(n_cols == 1)   // a column vector
      {
      arma_extra_debug_print("subview::extract(): copying col (going across rows)");
      
      // in.colptr(0) the first column of the subview, taking into account any row offset
      arrayops::copy( out.memptr(), in.colptr(0), n_rows );
      }
    else   // a row vector (possibly empty)
      {
      arma_extra_debug_print("subview::extract(): copying row (going across columns)");
      
      const Mat<eT>& X = in.m;
      
      eT* out_mem = out.memptr();
      
      const uword row       = in.aux_row1;
      const uword start_col = in.aux_col1;
      
      uword i,j;
      
      for(i=0, j=1; j < n_cols; i+=2, j+=2)
        {
        const eT tmp1 = X.at(row, start_col+i);
        const eT tmp2 = X.at(row, start_col+j);
        
        out_mem[i] = tmp1;
        out_mem[j] = tmp2;
        }
      
      if(i < n_cols)
        {
        out_mem[i] = X.at(row, start_col+i);
        }
      }
    }
  else   // general submatrix
    {
    arma_extra_debug_print("subview::extract(): general submatrix");
    
    for(uword col=0; col < n_cols; ++col)
      {
      arrayops::copy( out.colptr(col), in.colptr(col), n_rows );
      }
    }
  }



//! X += Y.submat(...)
template<typename eT>
inline
void
subview<eT>::plus_inplace(Mat<eT>& out, const subview<eT>& in)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(out, in, "addition");
  
  const uword n_rows = in.n_rows;
  const uword n_cols = in.n_cols;
  
  if(n_rows == 1)
    {
    eT* out_mem = out.memptr();
    
    const Mat<eT>& X = in.m;
    
    const uword row       = in.aux_row1;
    const uword start_col = in.aux_col1;
    
    uword i,j;
    for(i=0, j=1; j < n_cols; i+=2, j+=2)
      {
      const eT tmp1 = X.at(row, start_col+i);
      const eT tmp2 = X.at(row, start_col+j);
        
      out_mem[i] += tmp1;
      out_mem[j] += tmp2;
      }
    
    if(i < n_cols)
      {
      out_mem[i] += X.at(row, start_col+i);
      }
    }
  else
    {
    for(uword col=0; col < n_cols; ++col)
      {
      arrayops::inplace_plus(out.colptr(col), in.colptr(col), n_rows);
      }
    }
  }



//! X -= Y.submat(...)
template<typename eT>
inline
void
subview<eT>::minus_inplace(Mat<eT>& out, const subview<eT>& in)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(out, in, "subtraction");
  
  const uword n_rows = in.n_rows;
  const uword n_cols = in.n_cols;
  
  if(n_rows == 1)
    {
    eT* out_mem = out.memptr();
    
    const Mat<eT>& X = in.m;
    
    const uword row       = in.aux_row1;
    const uword start_col = in.aux_col1;
    
    uword i,j;
    for(i=0, j=1; j < n_cols; i+=2, j+=2)
      {
      const eT tmp1 = X.at(row, start_col+i);
      const eT tmp2 = X.at(row, start_col+j);
        
      out_mem[i] -= tmp1;
      out_mem[j] -= tmp2;
      }
    
    if(i < n_cols)
      {
      out_mem[i] -= X.at(row, start_col+i);
      }
    }
  else
    {
    for(uword col=0; col < n_cols; ++col)
      {
      arrayops::inplace_minus(out.colptr(col), in.colptr(col), n_rows);
      }
    }
  }



//! X %= Y.submat(...)
template<typename eT>
inline
void
subview<eT>::schur_inplace(Mat<eT>& out, const subview<eT>& in)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(out, in, "element-wise multiplication");
  
  const uword n_rows = in.n_rows;
  const uword n_cols = in.n_cols;
  
  if(n_rows == 1)
    {
    eT* out_mem = out.memptr();
    
    const Mat<eT>& X = in.m;
    
    const uword row       = in.aux_row1;
    const uword start_col = in.aux_col1;
    
    uword i,j;
    for(i=0, j=1; j < n_cols; i+=2, j+=2)
      {
      const eT tmp1 = X.at(row, start_col+i);
      const eT tmp2 = X.at(row, start_col+j);
        
      out_mem[i] *= tmp1;
      out_mem[j] *= tmp2;
      }
    
    if(i < n_cols)
      {
      out_mem[i] *= X.at(row, start_col+i);
      }
    }
  else
    {
    for(uword col=0; col < n_cols; ++col)
      {
      arrayops::inplace_mul(out.colptr(col), in.colptr(col), n_rows);
      }
    }
  }



//! X /= Y.submat(...)
template<typename eT>
inline
void
subview<eT>::div_inplace(Mat<eT>& out, const subview<eT>& in)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(out, in, "element-wise division");
  
  const uword n_rows = in.n_rows;
  const uword n_cols = in.n_cols;
  
  if(n_rows == 1)
    {
    eT* out_mem = out.memptr();
    
    const Mat<eT>& X = in.m;
    
    const uword row       = in.aux_row1;
    const uword start_col = in.aux_col1;
    
    uword i,j;
    for(i=0, j=1; j < n_cols; i+=2, j+=2)
      {
      const eT tmp1 = X.at(row, start_col+i);
      const eT tmp2 = X.at(row, start_col+j);
        
      out_mem[i] /= tmp1;
      out_mem[j] /= tmp2;
      }
    
    if(i < n_cols)
      {
      out_mem[i] /= X.at(row, start_col+i);
      }
    }
  else
    {
    for(uword col=0; col < n_cols; ++col)
      {
      arrayops::inplace_div(out.colptr(col), in.colptr(col), n_rows);
      }
    }
  }



//! creation of subview (row vector)
template<typename eT>
inline
subview_row<eT>
subview<eT>::row(const uword row_num)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( row_num >= n_rows, "subview::row(): out of bounds" );
  
  const uword base_row = aux_row1 + row_num;
  
  return subview_row<eT>(m, base_row, aux_col1, n_cols);
  }



//! creation of subview (row vector)
template<typename eT>
inline
const subview_row<eT>
subview<eT>::row(const uword row_num) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( row_num >= n_rows, "subview::row(): out of bounds" );
  
  const uword base_row = aux_row1 + row_num;
  
  return subview_row<eT>(m, base_row, aux_col1, n_cols);
  }



template<typename eT>
inline
subview_row<eT>
subview<eT>::operator()(const uword row_num, const span& col_span)
  {
  arma_extra_debug_sigprint();
  
  const bool col_all = col_span.whole;
  
  const uword local_n_cols = n_cols;
  
  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;
  
  const uword base_col1     = aux_col1 + in_col1;  
  const uword base_row      = aux_row1 + row_num;
  
  arma_debug_check
    (
    (row_num >= n_rows)
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "subview::operator(): indices out of bounds or incorrectly used"
    );
  
  return subview_row<eT>(m, base_row, base_col1, submat_n_cols);
  }



template<typename eT>
inline
const subview_row<eT>
subview<eT>::operator()(const uword row_num, const span& col_span) const
  {
  arma_extra_debug_sigprint();
  
  const bool col_all = col_span.whole;
  
  const uword local_n_cols = n_cols;
  
  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;
  
  const uword base_col1     = aux_col1 + in_col1;
  const uword base_row      = aux_row1 + row_num;
  
  arma_debug_check
    (
    (row_num >= n_rows)
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "subview::operator(): indices out of bounds or incorrectly used"
    );
  
  return subview_row<eT>(m, base_row, base_col1, submat_n_cols);
  }



//! creation of subview (column vector)
template<typename eT>
inline
subview_col<eT>
subview<eT>::col(const uword col_num)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( col_num >= n_cols, "subview::col(): out of bounds");
  
  const uword base_col = aux_col1 + col_num;
  
  return subview_col<eT>(m, base_col, aux_row1, n_rows);
  }



//! creation of subview (column vector)
template<typename eT>
inline
const subview_col<eT>
subview<eT>::col(const uword col_num) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( col_num >= n_cols, "subview::col(): out of bounds");
  
  const uword base_col = aux_col1 + col_num;
  
  return subview_col<eT>(m, base_col, aux_row1, n_rows);
  }



template<typename eT>
inline
subview_col<eT>
subview<eT>::operator()(const span& row_span, const uword col_num)
  {
  arma_extra_debug_sigprint();
  
  const bool row_all = row_span.whole;
  
  const uword local_n_rows = n_rows;
  
  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;
  
  const uword base_row1       = aux_row1 + in_row1;  
  const uword base_col        = aux_col1 + col_num;
  
  arma_debug_check
    (
    (col_num >= n_cols)
    ||
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ,
    "subview::operator(): indices out of bounds or incorrectly used"
    );
  
  return subview_col<eT>(m, base_col, base_row1, submat_n_rows);
  }



template<typename eT>
inline
const subview_col<eT>
subview<eT>::operator()(const span& row_span, const uword col_num) const
  {
  arma_extra_debug_sigprint();
  
  const bool row_all = row_span.whole;
  
  const uword local_n_rows = n_rows;
  
  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;
  
  const uword base_row1       = aux_row1 + in_row1;
  const uword base_col        = aux_col1 + col_num;
  
  arma_debug_check
    (
    (col_num >= n_cols)
    ||
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ,
    "subview::operator(): indices out of bounds or incorrectly used"
    );
  
  return subview_col<eT>(m, base_col, base_row1, submat_n_rows);
  }



//! create a Col object which uses memory from an existing matrix object.
//! this approach is currently not alias safe
//! and does not take into account that the parent matrix object could be deleted.
//! if deleted memory is accessed by the created Col object,
//! it will cause memory corruption and/or a crash
template<typename eT>
inline
Col<eT>
subview<eT>::unsafe_col(const uword col_num)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( col_num >= n_cols, "subview::unsafe_col(): out of bounds");
  
  return Col<eT>(colptr(col_num), n_rows, false, true);
  }



//! create a Col object which uses memory from an existing matrix object.
//! this approach is currently not alias safe
//! and does not take into account that the parent matrix object could be deleted.
//! if deleted memory is accessed by the created Col object,
//! it will cause memory corruption and/or a crash
template<typename eT>
inline
const Col<eT>
subview<eT>::unsafe_col(const uword col_num) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( col_num >= n_cols, "subview::unsafe_col(): out of bounds");
  
  return Col<eT>(const_cast<eT*>(colptr(col_num)), n_rows, false, true);
  }



//! creation of subview (submatrix comprised of specified row vectors)
template<typename eT>
inline
subview<eT>
subview<eT>::rows(const uword in_row1, const uword in_row2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "subview::rows(): indices out of bounds or incorrectly used"
    );
  
  const uword subview_n_rows = in_row2 - in_row1 + 1;
  const uword base_row1 = aux_row1 + in_row1;
  
  return subview<eT>(m, base_row1, aux_col1, subview_n_rows, n_cols );
  }



//! creation of subview (submatrix comprised of specified row vectors)
template<typename eT>
inline
const subview<eT>
subview<eT>::rows(const uword in_row1, const uword in_row2) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "subview::rows(): indices out of bounds or incorrectly used"
    );
  
  const uword subview_n_rows = in_row2 - in_row1 + 1;
  const uword base_row1 = aux_row1 + in_row1;
  
  return subview<eT>(m, base_row1, aux_col1, subview_n_rows, n_cols );
  }



//! creation of subview (submatrix comprised of specified column vectors)
template<typename eT>
inline
subview<eT>
subview<eT>::cols(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "subview::cols(): indices out of bounds or incorrectly used"
    );
  
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  const uword base_col1 = aux_col1 + in_col1;
  
  return subview<eT>(m, aux_row1, base_col1, n_rows, subview_n_cols);
  }



//! creation of subview (submatrix comprised of specified column vectors)
template<typename eT>
inline
const subview<eT>
subview<eT>::cols(const uword in_col1, const uword in_col2) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "subview::cols(): indices out of bounds or incorrectly used"
    );
  
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  const uword base_col1 = aux_col1 + in_col1;
  
  return subview<eT>(m, aux_row1, base_col1, n_rows, subview_n_cols);
  }



//! creation of subview (submatrix)
template<typename eT>
inline
subview<eT>
subview<eT>::submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check
    (
    (in_row1 > in_row2) || (in_col1 >  in_col2) || (in_row2 >= n_rows) || (in_col2 >= n_cols),
    "subview::submat(): indices out of bounds or incorrectly used"
    );
  
  const uword subview_n_rows = in_row2 - in_row1 + 1;
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  
  const uword base_row1 = aux_row1 + in_row1;
  const uword base_col1 = aux_col1 + in_col1;
  
  return subview<eT>(m, base_row1, base_col1, subview_n_rows, subview_n_cols);
  }



//! creation of subview (generic submatrix)
template<typename eT>
inline
const subview<eT>
subview<eT>::submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check
    (
    (in_row1 > in_row2) || (in_col1 >  in_col2) || (in_row2 >= n_rows) || (in_col2 >= n_cols),
    "subview::submat(): indices out of bounds or incorrectly used"
    );
  
  const uword subview_n_rows = in_row2 - in_row1 + 1;
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  
  const uword base_row1 = aux_row1 + in_row1;
  const uword base_col1 = aux_col1 + in_col1;
  
  return subview<eT>(m, base_row1, base_col1, subview_n_rows, subview_n_cols);
  }



//! creation of subview (submatrix)
template<typename eT>
inline
subview<eT>
subview<eT>::submat(const span& row_span, const span& col_span)
  {
  arma_extra_debug_sigprint();
  
  const bool row_all = row_span.whole;
  const bool col_all = col_span.whole;
  
  const uword local_n_rows = n_rows;
  const uword local_n_cols = n_cols;
  
  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;
  
  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;
  
  arma_debug_check
    (
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "subview::submat(): indices out of bounds or incorrectly used"
    );
  
  const uword base_row1 = aux_row1 + in_row1;
  const uword base_col1 = aux_col1 + in_col1;
  
  return subview<eT>(m, base_row1, base_col1, submat_n_rows, submat_n_cols);
  }



//! creation of subview (generic submatrix)
template<typename eT>
inline
const subview<eT>
subview<eT>::submat(const span& row_span, const span& col_span) const
  {
  arma_extra_debug_sigprint();
  
  const bool row_all = row_span.whole;
  const bool col_all = col_span.whole;
  
  const uword local_n_rows = n_rows;
  const uword local_n_cols = n_cols;
  
  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;
  
  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;
  
  arma_debug_check
    (
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "subview::submat(): indices out of bounds or incorrectly used"
    );
  
  const uword base_row1 = aux_row1 + in_row1;
  const uword base_col1 = aux_col1 + in_col1;
  
  return subview<eT>(m, base_row1, base_col1, submat_n_rows, submat_n_cols);
  }



template<typename eT>
inline
subview<eT>
subview<eT>::operator()(const span& row_span, const span& col_span)
  {
  arma_extra_debug_sigprint();
  
  return (*this).submat(row_span, col_span);
  }



template<typename eT>
inline
const subview<eT>
subview<eT>::operator()(const span& row_span, const span& col_span) const
  {
  arma_extra_debug_sigprint();
  
  return (*this).submat(row_span, col_span);
  }



template<typename eT>
inline
subview_each1< subview<eT>, 0 >
subview<eT>::each_col()
  {
  arma_extra_debug_sigprint();
  
  return subview_each1< subview<eT>, 0 >(*this);
  }



template<typename eT>
inline
subview_each1< subview<eT>, 1 >
subview<eT>::each_row()
  {
  arma_extra_debug_sigprint();
  
  return subview_each1< subview<eT>, 1 >(*this);
  }



template<typename eT>
template<typename T1>
inline
subview_each2< subview<eT>, 0, T1 >
subview<eT>::each_col(const Base<uword,T1>& indices)
  {
  arma_extra_debug_sigprint();
  
  return subview_each2< subview<eT>, 0, T1 >(*this, indices);
  }



template<typename eT>
template<typename T1>
inline
subview_each2< subview<eT>, 1, T1 >
subview<eT>::each_row(const Base<uword,T1>& indices)
  {
  arma_extra_debug_sigprint();
  
  return subview_each2< subview<eT>, 1, T1 >(*this, indices);
  }



//! creation of diagview (diagonal)
template<typename eT>
inline
diagview<eT>
subview<eT>::diag(const sword in_id)
  {
  arma_extra_debug_sigprint();
  
  const uword row_offset = (in_id < 0) ? uword(-in_id) : 0;
  const uword col_offset = (in_id > 0) ? uword( in_id) : 0;
  
  arma_debug_check
    (
    ((row_offset > 0) && (row_offset >= n_rows)) || ((col_offset > 0) && (col_offset >= n_cols)),
    "subview::diag(): requested diagonal out of bounds"
    );
  
  const uword len = (std::min)(n_rows - row_offset, n_cols - col_offset);
  
  const uword base_row_offset = aux_row1 + row_offset;
  const uword base_col_offset = aux_col1 + col_offset;
  
  return diagview<eT>(m, base_row_offset, base_col_offset, len);
  }



//! creation of diagview (diagonal)
template<typename eT>
inline
const diagview<eT>
subview<eT>::diag(const sword in_id) const
  {
  arma_extra_debug_sigprint();
  
  const uword row_offset = (in_id < 0) ? -in_id : 0;
  const uword col_offset = (in_id > 0) ?  in_id : 0;
  
  arma_debug_check
    (
    ((row_offset > 0) && (row_offset >= n_rows)) || ((col_offset > 0) && (col_offset >= n_cols)),
    "subview::diag(): requested diagonal out of bounds"
    );
  
  const uword len = (std::min)(n_rows - row_offset, n_cols - col_offset);
  
  const uword base_row_offset = aux_row1 + row_offset;
  const uword base_col_offset = aux_col1 + col_offset;
  
  return diagview<eT>(m, base_row_offset, base_col_offset, len);
  }



template<typename eT>
inline
void
subview<eT>::swap_rows(const uword in_row1, const uword in_row2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check
    (
    (in_row1 >= n_rows) || (in_row2 >= n_rows),
    "subview::swap_rows(): out of bounds"
    );
  
  eT* mem = (const_cast< Mat<eT>& >(m)).memptr();
  
  if(n_elem > 0)
    {
    const uword m_n_rows = m.n_rows;
    
    for(uword ucol=0; ucol < n_cols; ++ucol)
      {
      const uword offset = (aux_col1 + ucol) * m_n_rows;
      const uword pos1   = aux_row1 + in_row1 + offset;
      const uword pos2   = aux_row1 + in_row2 + offset;
      
      std::swap( access::rw(mem[pos1]), access::rw(mem[pos2]) );
      }
    }
  }



template<typename eT>
inline
void
subview<eT>::swap_cols(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check
    (
    (in_col1 >= n_cols) || (in_col2 >= n_cols),
    "subview::swap_cols(): out of bounds"
    );
  
  if(n_elem > 0)
    {
    eT* ptr1 = colptr(in_col1);
    eT* ptr2 = colptr(in_col2);
    
    for(uword urow=0; urow < n_rows; ++urow)
      {
      std::swap( ptr1[urow], ptr2[urow] );
      }
    }
  }



// template<typename eT>
// inline
// subview<eT>::iter::iter(const subview<eT>& S)
//   : mem       (S.m.mem)
//   , n_rows    (S.m.n_rows)
//   , row_start (S.aux_row1)
//   , row_end_p1(row_start + S.n_rows)
//   , row       (row_start)
//   , col       (S.aux_col1)
//   , i         (row + col*n_rows)
//   {
//   arma_extra_debug_sigprint();
//   }
// 
// 
// 
// template<typename eT>
// arma_inline
// eT
// subview<eT>::iter::operator*() const
//   {
//   return mem[i];
//   }
// 
// 
// 
// template<typename eT>
// inline
// void
// subview<eT>::iter::operator++()
//   {
//   ++row;
//   
//   if(row < row_end_p1)
//     {
//     ++i;
//     }
//   else
//     {
//     row = row_start;
//     ++col;
//     
//     i = row + col*n_rows;
//     }
//   }
// 
// 
// 
// template<typename eT>
// inline
// void
// subview<eT>::iter::operator++(int)
//   {
//   operator++();
//   }



//
//
//



template<typename eT>
inline
subview_col<eT>::subview_col(const Mat<eT>& in_m, const uword in_col)
  : subview<eT>(in_m, 0, in_col, in_m.n_rows, 1)
  , colmem(subview<eT>::colptr(0)) 
  {
  arma_extra_debug_sigprint();
  }



template<typename eT>
inline
subview_col<eT>::subview_col(const Mat<eT>& in_m, const uword in_col, const uword in_row1, const uword in_n_rows)
  : subview<eT>(in_m, in_row1, in_col, in_n_rows, 1)
  , colmem(subview<eT>::colptr(0)) 
  {
  arma_extra_debug_sigprint();
  }



template<typename eT>
inline
void
subview_col<eT>::operator=(const subview<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  subview<eT>::operator=(X);
  }



template<typename eT>
inline
void
subview_col<eT>::operator=(const subview_col<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  subview<eT>::operator=(X); // interprets 'subview_col' as 'subview'
  }



template<typename eT>
inline
void
subview_col<eT>::operator=(const eT val)
  {
  arma_extra_debug_sigprint();
  
  if(subview<eT>::n_elem != 1)
    {
    arma_debug_assert_same_size(subview<eT>::n_rows, subview<eT>::n_cols, 1, 1, "copy into submatrix");
    }
  
  access::rw( colmem[0] ) = val;
  }



template<typename eT>
template<typename T1>
inline
void
subview_col<eT>::operator=(const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  subview<eT>::operator=(X);
  }



template<typename eT>
template<typename T1, typename gen_type>
inline
typename enable_if2< is_same_type<typename T1::elem_type, eT>::value, void>::result
subview_col<eT>::operator= (const Gen<T1,gen_type>& in)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(subview<eT>::n_rows, uword(1), in.n_rows, (in.is_col ? uword(1) : in.n_cols), "copy into submatrix");
  
  in.apply(*this);
  }



template<typename eT>
arma_inline
const Op<subview_col<eT>,op_htrans>
subview_col<eT>::t() const
  {
  return Op<subview_col<eT>,op_htrans>(*this);
  }



template<typename eT>
arma_inline
const Op<subview_col<eT>,op_htrans>
subview_col<eT>::ht() const
  {
  return Op<subview_col<eT>,op_htrans>(*this);
  }



template<typename eT>
arma_inline
const Op<subview_col<eT>,op_strans>
subview_col<eT>::st() const
  {
  return Op<subview_col<eT>,op_strans>(*this);
  }



template<typename eT>
inline
void
subview_col<eT>::fill(const eT val)
  {
  arma_extra_debug_sigprint();
  
  arrayops::inplace_set( access::rwp(colmem), val, subview<eT>::n_rows );
  }



template<typename eT>
inline
void
subview_col<eT>::zeros()
  {
  arma_extra_debug_sigprint();
  
  arrayops::fill_zeros( access::rwp(colmem), subview<eT>::n_rows );
  }



template<typename eT>
inline
void
subview_col<eT>::ones()
  {
  arma_extra_debug_sigprint();
  
  arrayops::inplace_set( access::rwp(colmem), eT(1), subview<eT>::n_rows );
  }



template<typename eT>
arma_inline
eT
subview_col<eT>::at_alt(const uword ii) const
  {
  const eT* colmem_aligned = colmem;
  memory::mark_as_aligned(colmem_aligned);
  
  return colmem_aligned[ii];
  }



template<typename eT>
arma_inline
eT&
subview_col<eT>::operator[](const uword ii)
  {
  return access::rw( colmem[ii] );
  }



template<typename eT>
arma_inline
eT
subview_col<eT>::operator[](const uword ii) const
  {
  return colmem[ii];
  }



template<typename eT>
inline
eT&
subview_col<eT>::operator()(const uword ii)
  {
  arma_debug_check( (ii >= subview<eT>::n_elem), "subview::operator(): index out of bounds");
    
  return access::rw( colmem[ii] );
  }



template<typename eT>
inline
eT
subview_col<eT>::operator()(const uword ii) const
  {
  arma_debug_check( (ii >= subview<eT>::n_elem), "subview::operator(): index out of bounds");
  
  return colmem[ii];
  }



template<typename eT>
inline
eT&
subview_col<eT>::operator()(const uword in_row, const uword in_col)
  {
  arma_debug_check( ((in_row >= subview<eT>::n_rows) || (in_col > 0)), "subview::operator(): index out of bounds");
  
  return access::rw( colmem[in_row] );
  }



template<typename eT>
inline
eT
subview_col<eT>::operator()(const uword in_row, const uword in_col) const
  {
  arma_debug_check( ((in_row >= subview<eT>::n_rows) || (in_col > 0)), "subview::operator(): index out of bounds");
  
  return colmem[in_row];
  }



template<typename eT>
inline
eT&
subview_col<eT>::at(const uword in_row, const uword)
  {
  return access::rw( colmem[in_row] );
  }



template<typename eT>
inline
eT
subview_col<eT>::at(const uword in_row, const uword) const
  {
  return colmem[in_row];
  }



template<typename eT>
arma_inline
eT*
subview_col<eT>::colptr(const uword)
  {
  return const_cast<eT*>(colmem);
  }
  
  
template<typename eT>
arma_inline
const eT*
subview_col<eT>::colptr(const uword) const
  {
  return colmem;
  }


template<typename eT>
inline
subview_col<eT>
subview_col<eT>::rows(const uword in_row1, const uword in_row2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_row1 > in_row2) || (in_row2 >= subview<eT>::n_rows) ), "subview_col::rows(): indices out of bounds or incorrectly used");
  
  const uword subview_n_rows = in_row2 - in_row1 + 1;
  
  const uword base_row1 = this->aux_row1 + in_row1;
  
  return subview_col<eT>(this->m, this->aux_col1, base_row1, subview_n_rows);
  }



template<typename eT>
inline
const subview_col<eT>
subview_col<eT>::rows(const uword in_row1, const uword in_row2) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_row1 > in_row2) || (in_row2 >= subview<eT>::n_rows) ), "subview_col::rows(): indices out of bounds or incorrectly used");
  
  const uword subview_n_rows = in_row2 - in_row1 + 1;
  
  const uword base_row1 = this->aux_row1 + in_row1;
  
  return subview_col<eT>(this->m, this->aux_col1, base_row1, subview_n_rows);
  }



template<typename eT>
inline
subview_col<eT>
subview_col<eT>::subvec(const uword in_row1, const uword in_row2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_row1 > in_row2) || (in_row2 >= subview<eT>::n_rows) ), "subview_col::subvec(): indices out of bounds or incorrectly used");
  
  const uword subview_n_rows = in_row2 - in_row1 + 1;
  
  const uword base_row1 = this->aux_row1 + in_row1;
  
  return subview_col<eT>(this->m, this->aux_col1, base_row1, subview_n_rows);
  }



template<typename eT>
inline
const subview_col<eT>
subview_col<eT>::subvec(const uword in_row1, const uword in_row2) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_row1 > in_row2) || (in_row2 >= subview<eT>::n_rows) ), "subview_col::subvec(): indices out of bounds or incorrectly used");
  
  const uword subview_n_rows = in_row2 - in_row1 + 1;
  
  const uword base_row1 = this->aux_row1 + in_row1;
  
  return subview_col<eT>(this->m, this->aux_col1, base_row1, subview_n_rows);
  }



template<typename eT>
inline
subview_col<eT>
subview_col<eT>::head(const uword N)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > subview<eT>::n_rows), "subview_col::head(): size out of bounds");
  
  return subview_col<eT>(this->m, this->aux_col1, this->aux_row1, N);
  }



template<typename eT>
inline
const subview_col<eT>
subview_col<eT>::head(const uword N) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > subview<eT>::n_rows), "subview_col::head(): size out of bounds");
  
  return subview_col<eT>(this->m, this->aux_col1, this->aux_row1, N);
  }



template<typename eT>
inline
subview_col<eT>
subview_col<eT>::tail(const uword N)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > subview<eT>::n_rows), "subview_col::tail(): size out of bounds");
  
  const uword start_row = subview<eT>::aux_row1 + subview<eT>::n_rows - N;
  
  return subview_col<eT>(this->m, this->aux_col1, start_row, N);
  }



template<typename eT>
inline
const subview_col<eT>
subview_col<eT>::tail(const uword N) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > subview<eT>::n_rows), "subview_col::tail(): size out of bounds");
  
  const uword start_row = subview<eT>::aux_row1 + subview<eT>::n_rows - N;
  
  return subview_col<eT>(this->m, this->aux_col1, start_row, N);
  }



//
//
//



template<typename eT>
inline
subview_row<eT>::subview_row(const Mat<eT>& in_m, const uword in_row)
  : subview<eT>(in_m, in_row, 0, 1, in_m.n_cols)
  {
  arma_extra_debug_sigprint();
  }



template<typename eT>
inline
subview_row<eT>::subview_row(const Mat<eT>& in_m, const uword in_row, const uword in_col1, const uword in_n_cols)
  : subview<eT>(in_m, in_row, in_col1, 1, in_n_cols)
  {
  arma_extra_debug_sigprint();
  }



template<typename eT>
inline
void
subview_row<eT>::operator=(const subview<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  subview<eT>::operator=(X);
  }



template<typename eT>
inline
void
subview_row<eT>::operator=(const subview_row<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  subview<eT>::operator=(X); // interprets 'subview_row' as 'subview'
  }



template<typename eT>
inline
void
subview_row<eT>::operator=(const eT val)
  {
  arma_extra_debug_sigprint();
  
  subview<eT>::operator=(val); // interprets 'subview_row' as 'subview'
  }



template<typename eT>
template<typename T1>
inline
void
subview_row<eT>::operator=(const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  subview<eT>::operator=(X);
  }



template<typename eT>
template<typename T1, typename gen_type>
inline
typename enable_if2< is_same_type<typename T1::elem_type, eT>::value, void>::result
subview_row<eT>::operator= (const Gen<T1,gen_type>& in)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(uword(1), subview<eT>::n_cols, (in.is_row ? uword(1) : in.n_rows), in.n_cols, "copy into submatrix");
  
  in.apply(*this);
  }



template<typename eT>
arma_inline
const Op<subview_row<eT>,op_htrans>
subview_row<eT>::t() const
  {
  return Op<subview_row<eT>,op_htrans>(*this);
  }



template<typename eT>
arma_inline
const Op<subview_row<eT>,op_htrans>
subview_row<eT>::ht() const
  {
  return Op<subview_row<eT>,op_htrans>(*this);
  }



template<typename eT>
arma_inline
const Op<subview_row<eT>,op_strans>
subview_row<eT>::st() const
  {
  return Op<subview_row<eT>,op_strans>(*this);
  }



template<typename eT>
inline
eT
subview_row<eT>::at_alt(const uword ii) const
  {
  const uword index = (ii + (subview<eT>::aux_col1))*(subview<eT>::m).n_rows + (subview<eT>::aux_row1);
  
  return subview<eT>::m.mem[index];
  }



template<typename eT>
inline
eT&
subview_row<eT>::operator[](const uword ii)
  {
  const uword index = (ii + (subview<eT>::aux_col1))*(subview<eT>::m).n_rows + (subview<eT>::aux_row1);
  
  return access::rw( (const_cast< Mat<eT>& >(subview<eT>::m)).mem[index] );
  }



template<typename eT>
inline
eT
subview_row<eT>::operator[](const uword ii) const
  {
  const uword index = (ii + (subview<eT>::aux_col1))*(subview<eT>::m).n_rows + (subview<eT>::aux_row1);
  
  return subview<eT>::m.mem[index];
  }



template<typename eT>
inline
eT&
subview_row<eT>::operator()(const uword ii)
  {
  arma_debug_check( (ii >= subview<eT>::n_elem), "subview::operator(): index out of bounds");
    
  const uword index = (ii + (subview<eT>::aux_col1))*(subview<eT>::m).n_rows + (subview<eT>::aux_row1);
  
  return access::rw( (const_cast< Mat<eT>& >(subview<eT>::m)).mem[index] );
  }



template<typename eT>
inline
eT
subview_row<eT>::operator()(const uword ii) const
  {
  arma_debug_check( (ii >= subview<eT>::n_elem), "subview::operator(): index out of bounds");
  
  const uword index = (ii + (subview<eT>::aux_col1))*(subview<eT>::m).n_rows + (subview<eT>::aux_row1);
  
  return subview<eT>::m.mem[index];
  }



template<typename eT>
inline
eT&
subview_row<eT>::operator()(const uword in_row, const uword in_col)
  {
  arma_debug_check( ((in_row > 0) || (in_col >= subview<eT>::n_cols)), "subview::operator(): index out of bounds");
  
  const uword index = (in_col + (subview<eT>::aux_col1))*(subview<eT>::m).n_rows + (subview<eT>::aux_row1);
  
  return access::rw( (const_cast< Mat<eT>& >(subview<eT>::m)).mem[index] );
  }



template<typename eT>
inline
eT
subview_row<eT>::operator()(const uword in_row, const uword in_col) const
  {
  arma_debug_check( ((in_row > 0) || (in_col >= subview<eT>::n_cols)), "subview::operator(): index out of bounds");
  
  const uword index = (in_col + (subview<eT>::aux_col1))*(subview<eT>::m).n_rows + (subview<eT>::aux_row1);
  
  return subview<eT>::m.mem[index];
  }



template<typename eT>
inline
eT&
subview_row<eT>::at(const uword, const uword in_col)
  {
  const uword index = (in_col + (subview<eT>::aux_col1))*(subview<eT>::m).n_rows + (subview<eT>::aux_row1);
  
  return access::rw( (const_cast< Mat<eT>& >(subview<eT>::m)).mem[index] );
  }



template<typename eT>
inline
eT
subview_row<eT>::at(const uword, const uword in_col) const
  {
  const uword index = (in_col + (subview<eT>::aux_col1))*(subview<eT>::m).n_rows + (subview<eT>::aux_row1);
  
  return subview<eT>::m.mem[index];
  }



template<typename eT>
inline
subview_row<eT>
subview_row<eT>::cols(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_col1 > in_col2) || (in_col2 >= subview<eT>::n_cols) ), "subview_row::cols(): indices out of bounds or incorrectly used" );
  
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  
  const uword base_col1 = this->aux_col1 + in_col1;
  
  return subview_row<eT>(this->m, this->aux_row1, base_col1, subview_n_cols);
  }



template<typename eT>
inline
const subview_row<eT>
subview_row<eT>::cols(const uword in_col1, const uword in_col2) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_col1 > in_col2) || (in_col2 >= subview<eT>::n_cols) ), "subview_row::cols(): indices out of bounds or incorrectly used");
  
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  
  const uword base_col1 = this->aux_col1 + in_col1;
  
  return subview_row<eT>(this->m, this->aux_row1, base_col1, subview_n_cols);
  }



template<typename eT>
inline
subview_row<eT>
subview_row<eT>::subvec(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_col1 > in_col2) || (in_col2 >= subview<eT>::n_cols) ), "subview_row::subvec(): indices out of bounds or incorrectly used");
  
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  
  const uword base_col1 = this->aux_col1 + in_col1;
  
  return subview_row<eT>(this->m, this->aux_row1, base_col1, subview_n_cols);
  }



template<typename eT>
inline
const subview_row<eT>
subview_row<eT>::subvec(const uword in_col1, const uword in_col2) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (in_col1 > in_col2) || (in_col2 >= subview<eT>::n_cols) ), "subview_row::subvec(): indices out of bounds or incorrectly used");
  
  const uword subview_n_cols = in_col2 - in_col1 + 1;
  
  const uword base_col1 = this->aux_col1 + in_col1;
  
  return subview_row<eT>(this->m, this->aux_row1, base_col1, subview_n_cols);
  }



template<typename eT>
inline
subview_row<eT>
subview_row<eT>::head(const uword N)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > subview<eT>::n_cols), "subview_row::head(): size out of bounds");
  
  return subview_row<eT>(this->m, this->aux_row1, this->aux_col1, N);
  }



template<typename eT>
inline
const subview_row<eT>
subview_row<eT>::head(const uword N) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > subview<eT>::n_cols), "subview_row::head(): size out of bounds");
  
  return subview_row<eT>(this->m, this->aux_row1, this->aux_col1, N);
  }



template<typename eT>
inline
subview_row<eT>
subview_row<eT>::tail(const uword N)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > subview<eT>::n_cols), "subview_row::tail(): size out of bounds");
  
  const uword start_col = subview<eT>::aux_col1 + subview<eT>::n_cols - N;
  
  return subview_row<eT>(this->m, this->aux_row1, start_col, N);
  }



template<typename eT>
inline
const subview_row<eT>
subview_row<eT>::tail(const uword N) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > subview<eT>::n_cols), "subview_row::tail(): size out of bounds");
  
  const uword start_col = subview<eT>::aux_col1 + subview<eT>::n_cols - N;
  
  return subview_row<eT>(this->m, this->aux_row1, start_col, N);
  }



//
//
//



template<typename eT>
inline
subview_row_strans<eT>::subview_row_strans(const subview_row<eT>& in_sv_row)
  : sv_row(in_sv_row       )
  , n_rows(in_sv_row.n_cols)
  , n_elem(in_sv_row.n_elem)
  {
  arma_extra_debug_sigprint();
  }



template<typename eT>
inline
void
subview_row_strans<eT>::extract(Mat<eT>& out) const
  {
  arma_extra_debug_sigprint();
  
  // NOTE: this function assumes that matrix 'out' has already been set to the correct size
  
  const Mat<eT>& X = sv_row.m;
  
  eT* out_mem = out.memptr();
  
  const uword row           = sv_row.aux_row1;
  const uword start_col     = sv_row.aux_col1;
  const uword sv_row_n_cols = sv_row.n_cols;
  
  uword ii,jj;
  
  for(ii=0, jj=1; jj < sv_row_n_cols; ii+=2, jj+=2)
    {
    const eT tmp1 = X.at(row, start_col+ii);
    const eT tmp2 = X.at(row, start_col+jj);
    
    out_mem[ii] = tmp1;
    out_mem[jj] = tmp2;
    }
  
  if(ii < sv_row_n_cols)
    {
    out_mem[ii] = X.at(row, start_col+ii);
    }
  }



template<typename eT>
inline
eT
subview_row_strans<eT>::at_alt(const uword ii) const
  {
  return sv_row[ii];
  }



template<typename eT>
inline
eT
subview_row_strans<eT>::operator[](const uword ii) const
  {
  return sv_row[ii];
  }



template<typename eT>
inline
eT
subview_row_strans<eT>::operator()(const uword ii) const
  {
  return sv_row(ii);
  }



template<typename eT>
inline
eT
subview_row_strans<eT>::operator()(const uword in_row, const uword in_col) const
  {
  return sv_row(in_col, in_row);  // deliberately swapped
  }



template<typename eT>
inline
eT
subview_row_strans<eT>::at(const uword in_row, const uword) const
  {
  return sv_row.at(0, in_row);  // deliberately swapped
  }



//
//
//



template<typename eT>
inline
subview_row_htrans<eT>::subview_row_htrans(const subview_row<eT>& in_sv_row)
  : sv_row(in_sv_row       )
  , n_rows(in_sv_row.n_cols)
  , n_elem(in_sv_row.n_elem)
  {
  arma_extra_debug_sigprint();
  }



template<typename eT>
inline
void
subview_row_htrans<eT>::extract(Mat<eT>& out) const
  {
  arma_extra_debug_sigprint();
  
  // NOTE: this function assumes that matrix 'out' has already been set to the correct size
  
  const Mat<eT>& X = sv_row.m;
  
  eT* out_mem = out.memptr();
  
  const uword row           = sv_row.aux_row1;
  const uword start_col     = sv_row.aux_col1;
  const uword sv_row_n_cols = sv_row.n_cols;
  
  for(uword ii=0; ii < sv_row_n_cols; ++ii)
    {
    out_mem[ii] = access::alt_conj( X.at(row, start_col+ii) );
    }
  }



template<typename eT>
inline
eT
subview_row_htrans<eT>::at_alt(const uword ii) const
  {
  return access::alt_conj( sv_row[ii] );
  }



template<typename eT>
inline
eT
subview_row_htrans<eT>::operator[](const uword ii) const
  {
  return access::alt_conj( sv_row[ii] );
  }



template<typename eT>
inline
eT
subview_row_htrans<eT>::operator()(const uword ii) const
  {
  return access::alt_conj( sv_row(ii) );
  }



template<typename eT>
inline
eT
subview_row_htrans<eT>::operator()(const uword in_row, const uword in_col) const
  {
  return access::alt_conj( sv_row(in_col, in_row) );  // deliberately swapped
  }



template<typename eT>
inline
eT
subview_row_htrans<eT>::at(const uword in_row, const uword) const
  {
  return access::alt_conj( sv_row.at(0, in_row) );  // deliberately swapped
  }



//! @}
