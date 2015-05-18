// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup subview_each
//! @{


//
//
// subview_each_common

template<typename parent, unsigned int mode>
inline
subview_each_common<parent,mode>::subview_each_common(parent& in_p)
  : p(in_p)
  {
  arma_extra_debug_sigprint();
  }



template<typename parent, unsigned int mode>
arma_inline
const Mat<typename parent::elem_type>&
subview_each_common<parent,mode>::get_mat_ref_helper(const Mat<typename parent::elem_type>& X) const
  {
  return X;
  }



template<typename parent, unsigned int mode>
arma_inline
const Mat<typename parent::elem_type>&
subview_each_common<parent,mode>::get_mat_ref_helper(const subview<typename parent::elem_type>& X) const
  {
  return X.m;
  }



template<typename parent, unsigned int mode>
arma_inline
const Mat<typename parent::elem_type>&
subview_each_common<parent,mode>::get_mat_ref() const
  {
  return get_mat_ref_helper(p);
  }



template<typename parent, unsigned int mode>
inline
void
subview_each_common<parent,mode>::check_size(const Mat<typename parent::elem_type>& A) const
  {
  if(arma_config::debug == true)
    {
    if(mode == 0)
      {
      if( (A.n_rows != p.n_rows) || (A.n_cols != 1) )
        {
        arma_stop( incompat_size_string(A) );
        }
      }
    else
      {
      if( (A.n_rows != 1) || (A.n_cols != p.n_cols) )
        {
        arma_stop( incompat_size_string(A) );
        }
      }
    }
  }



template<typename parent, unsigned int mode>
arma_cold
inline
const std::string
subview_each_common<parent,mode>::incompat_size_string(const Mat<typename parent::elem_type>& A) const
  {
  std::stringstream tmp;
  
  if(mode == 0)
    {
    tmp << "each_col(): incompatible size; expected " << p.n_rows << "x1" << ", got " << A.n_rows << 'x' << A.n_cols;
    }
  else
    {
    tmp << "each_row(): incompatible size; expected 1x" << p.n_cols << ", got " << A.n_rows << 'x' << A.n_cols;
    }
  
  return tmp.str();
  }
  


//
//
// subview_each1



template<typename parent, unsigned int mode>
inline
subview_each1<parent,mode>::~subview_each1()
  {
  arma_extra_debug_sigprint();
  }



template<typename parent, unsigned int mode>
inline
subview_each1<parent,mode>::subview_each1(parent& in_p)
  : subview_each_common<parent,mode>::subview_each_common(in_p)
  {
  arma_extra_debug_sigprint();
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent,mode>::operator= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  parent& p = subview_each_common<parent,mode>::p;
  
  const unwrap_check<T1> tmp( in.get_ref(), (*this).get_mat_ref() );
  const Mat<eT>& A     = tmp.M;
  
  subview_each_common<parent,mode>::check_size(A);
  
  const eT*   A_mem    = A.memptr();
  const uword p_n_rows = p.n_rows;
  const uword p_n_cols = p.n_cols;
  
  if(mode == 0) // each column
    {
    for(uword i=0; i < p_n_cols; ++i)
      {
      arrayops::copy( p.colptr(i), A_mem, p_n_rows );
      }
    }
  else // each row
    {
    for(uword i=0; i < p_n_cols; ++i)
      {
      arrayops::inplace_set( p.colptr(i), A_mem[i], p_n_rows);
      }
    }
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent,mode>::operator+= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  parent& p = subview_each_common<parent,mode>::p;
  
  const unwrap_check<T1> tmp( in.get_ref(), (*this).get_mat_ref() );
  const Mat<eT>& A     = tmp.M;
  
  subview_each_common<parent,mode>::check_size(A);
  
  const eT*   A_mem    = A.memptr();
  const uword p_n_rows = p.n_rows;
  const uword p_n_cols = p.n_cols;
    
  if(mode == 0) // each column
    {
    for(uword i=0; i < p_n_cols; ++i)
      {
      arrayops::inplace_plus( p.colptr(i), A_mem, p_n_rows );
      }
    }
  else // each row
    {
    for(uword i=0; i < p_n_cols; ++i)
      {
      arrayops::inplace_plus( p.colptr(i), A_mem[i], p_n_rows);
      }
    }
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent,mode>::operator-= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  parent& p = subview_each_common<parent,mode>::p;
  
  const unwrap_check<T1> tmp( in.get_ref(), (*this).get_mat_ref() );
  const Mat<eT>& A     = tmp.M;
  
  subview_each_common<parent,mode>::check_size(A);
  
  const eT*   A_mem    = A.memptr();
  const uword p_n_rows = p.n_rows;
  const uword p_n_cols = p.n_cols;
    
  if(mode == 0) // each column
    {
    for(uword i=0; i < p_n_cols; ++i)
      {
      arrayops::inplace_minus( p.colptr(i), A_mem, p_n_rows );
      }
    }
  else // each row
    {
    for(uword i=0; i < p_n_cols; ++i)
      {
      arrayops::inplace_minus( p.colptr(i), A_mem[i], p_n_rows);
      }
    }
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent,mode>::operator%= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  parent& p = subview_each_common<parent,mode>::p;
  
  const unwrap_check<T1> tmp( in.get_ref(), (*this).get_mat_ref() );
  const Mat<eT>& A     = tmp.M;
  
  subview_each_common<parent,mode>::check_size(A);
  
  const eT*   A_mem    = A.memptr();
  const uword p_n_rows = p.n_rows;
  const uword p_n_cols = p.n_cols;
  
  if(mode == 0) // each column
    {
    for(uword i=0; i < p_n_cols; ++i)
      {
      arrayops::inplace_mul( p.colptr(i), A_mem, p_n_rows );
      }
    }
  else // each row
    {
    for(uword i=0; i < p_n_cols; ++i)
      {
      arrayops::inplace_mul( p.colptr(i), A_mem[i], p_n_rows);
      }
    }
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent,mode>::operator/= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  parent& p = subview_each_common<parent,mode>::p;
  
  const unwrap_check<T1> tmp( in.get_ref(), (*this).get_mat_ref() );
  const Mat<eT>& A     = tmp.M;
  
  subview_each_common<parent,mode>::check_size(A);
  
  const eT*   A_mem    = A.memptr();
  const uword p_n_rows = p.n_rows;
  const uword p_n_cols = p.n_cols;
  
  if(mode == 0) // each column
    {
    for(uword i=0; i < p_n_cols; ++i)
      {
      arrayops::inplace_div( p.colptr(i), A_mem, p_n_rows );
      }
    }
  else // each row
    {
    for(uword i=0; i < p_n_cols; ++i)
      {
      arrayops::inplace_div( p.colptr(i), A_mem[i], p_n_rows);
      }
    }
  }



//
//
// subview_each2



template<typename parent, unsigned int mode, typename TB>
inline
subview_each2<parent,mode,TB>::~subview_each2()
  {
  arma_extra_debug_sigprint();
  }



template<typename parent, unsigned int mode, typename TB>
inline
subview_each2<parent,mode,TB>::subview_each2(parent& in_p, const Base<uword, TB>& in_indices)
  : subview_each_common<parent,mode>::subview_each_common(in_p)
  , base_indices(in_indices)
  {
  arma_extra_debug_sigprint();
  }



template<typename parent, unsigned int mode, typename TB>
inline
void
subview_each2<parent,mode,TB>::check_indices(const Mat<uword>& indices) const
  {
  if(mode == 0)
    {
    arma_debug_check( ((indices.is_vec() == false) && (indices.is_empty() == false)), "each_col(): list of indices must be a vector" );
    }
  else
    {
    arma_debug_check( ((indices.is_vec() == false) && (indices.is_empty() == false)), "each_row(): list of indices must be a vector" );
    }
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent,mode,TB>::operator= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  parent& p = subview_each_common<parent,mode>::p;
  
  const unwrap_check<T1> tmp( in.get_ref(), (*this).get_mat_ref() );
  const Mat<eT>& A     = tmp.M;
  
  subview_each_common<parent,mode>::check_size(A);
  
  const unwrap_check_mixed<TB> tmp_indices( base_indices.get_ref(), (*this).get_mat_ref() );
  const Mat<uword>& indices =  tmp_indices.M;
  
  check_indices(indices);
  
  const eT*   A_mem    = A.memptr();
  const uword p_n_rows = p.n_rows;
  const uword p_n_cols = p.n_cols;
  
  const uword* indices_mem = indices.memptr();
  const uword  N           = indices.n_elem;
  
  if(mode == 0) // each column
    {
    for(uword i=0; i < N; ++i)
      {
      const uword col = indices_mem[i];
      
      arma_debug_check( (col > p_n_cols), "each_col(): index out of bounds" );
      
      arrayops::copy( p.colptr(col), A_mem, p_n_rows );
      }
    }
  else // each row
    {
    for(uword i=0; i < N; ++i)
      {
      const uword row = indices_mem[i];
      
      arma_debug_check( (row > p_n_rows), "each_row(): index out of bounds" );
      
      for(uword col=0; col < p_n_cols; ++col)
        {
        p.at(row,col) = A_mem[col];
        }
      }
    }
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent,mode,TB>::operator+= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  parent& p = subview_each_common<parent,mode>::p;
  
  const unwrap_check<T1> tmp( in.get_ref(), (*this).get_mat_ref() );
  const Mat<eT>& A     = tmp.M;
  
  subview_each_common<parent,mode>::check_size(A);
  
  const unwrap_check_mixed<TB> tmp_indices( base_indices.get_ref(), (*this).get_mat_ref() );
  const Mat<uword>& indices =  tmp_indices.M;
  
  check_indices(indices);
  
  const eT*   A_mem    = A.memptr();
  const uword p_n_rows = p.n_rows;
  const uword p_n_cols = p.n_cols;
  
  const uword* indices_mem = indices.memptr();
  const uword  N           = indices.n_elem;
  
  if(mode == 0) // each column
    {
    for(uword i=0; i < N; ++i)
      {
      const uword col = indices_mem[i];
      
      arma_debug_check( (col > p_n_cols), "each_col(): index out of bounds" );
      
      arrayops::inplace_plus( p.colptr(col), A_mem, p_n_rows );
      }
    }
  else // each row
    {
    for(uword i=0; i < N; ++i)
      {
      const uword row = indices_mem[i];
      
      arma_debug_check( (row > p_n_rows), "each_row(): index out of bounds" );
      
      for(uword col=0; col < p_n_cols; ++col)
        {
        p.at(row,col) += A_mem[col];
        }
      }
    }
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent,mode,TB>::operator-= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  parent& p = subview_each_common<parent,mode>::p;
  
  const unwrap_check<T1> tmp( in.get_ref(), (*this).get_mat_ref() );
  const Mat<eT>& A     = tmp.M;
  
  subview_each_common<parent,mode>::check_size(A);
  
  const unwrap_check_mixed<TB> tmp_indices( base_indices.get_ref(), (*this).get_mat_ref() );
  const Mat<uword>& indices =  tmp_indices.M;
  
  check_indices(indices);
  
  const eT*   A_mem    = A.memptr();
  const uword p_n_rows = p.n_rows;
  const uword p_n_cols = p.n_cols;
  
  const uword* indices_mem = indices.memptr();
  const uword  N           = indices.n_elem;
  
  if(mode == 0) // each column
    {
    for(uword i=0; i < N; ++i)
      {
      const uword col = indices_mem[i];
      
      arma_debug_check( (col > p_n_cols), "each_col(): index out of bounds" );
      
      arrayops::inplace_minus( p.colptr(col), A_mem, p_n_rows );
      }
    }
  else // each row
    {
    for(uword i=0; i < N; ++i)
      {
      const uword row = indices_mem[i];
      
      arma_debug_check( (row > p_n_rows), "each_row(): index out of bounds" );
      
      for(uword col=0; col < p_n_cols; ++col)
        {
        p.at(row,col) -= A_mem[col];
        }
      }
    }
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent,mode,TB>::operator%= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  parent& p = subview_each_common<parent,mode>::p;
  
  const unwrap_check<T1> tmp( in.get_ref(), (*this).get_mat_ref() );
  const Mat<eT>& A     = tmp.M;
  
  subview_each_common<parent,mode>::check_size(A);
  
  const unwrap_check_mixed<TB> tmp_indices( base_indices.get_ref(), (*this).get_mat_ref() );
  const Mat<uword>& indices =  tmp_indices.M;
  
  check_indices(indices);
  
  const eT*   A_mem    = A.memptr();
  const uword p_n_rows = p.n_rows;
  const uword p_n_cols = p.n_cols;
  
  const uword* indices_mem = indices.memptr();
  const uword  N           = indices.n_elem;
  
  if(mode == 0) // each column
    {
    for(uword i=0; i < N; ++i)
      {
      const uword col = indices_mem[i];
      
      arma_debug_check( (col > p_n_cols), "each_col(): index out of bounds" );
      
      arrayops::inplace_mul( p.colptr(col), A_mem, p_n_rows );
      }
    }
  else // each row
    {
    for(uword i=0; i < N; ++i)
      {
      const uword row = indices_mem[i];
      
      arma_debug_check( (row > p_n_rows), "each_row(): index out of bounds" );
      
      for(uword col=0; col < p_n_cols; ++col)
        {
        p.at(row,col) *= A_mem[col];
        }
      }
    }
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent,mode,TB>::operator/= (const Base<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  parent& p = subview_each_common<parent,mode>::p;
  
  const unwrap_check<T1> tmp( in.get_ref(), (*this).get_mat_ref() );
  const Mat<eT>& A     = tmp.M;
  
  subview_each_common<parent,mode>::check_size(A);
  
  const unwrap_check_mixed<TB> tmp_indices( base_indices.get_ref(), (*this).get_mat_ref() );
  const Mat<uword>& indices =  tmp_indices.M;
  
  check_indices(indices);
  
  const eT*   A_mem    = A.memptr();
  const uword p_n_rows = p.n_rows;
  const uword p_n_cols = p.n_cols;
  
  const uword* indices_mem = indices.memptr();
  const uword  N           = indices.n_elem;
  
  if(mode == 0) // each column
    {
    for(uword i=0; i < N; ++i)
      {
      const uword col = indices_mem[i];
      
      arma_debug_check( (col > p_n_cols), "each_col(): index out of bounds" );
      
      arrayops::inplace_div( p.colptr(col), A_mem, p_n_rows );
      }
    }
  else // each row
    {
    for(uword i=0; i < N; ++i)
      {
      const uword row = indices_mem[i];
      
      arma_debug_check( (row > p_n_rows), "each_row(): index out of bounds" );
      
      for(uword col=0; col < p_n_cols; ++col)
        {
        p.at(row,col) /= A_mem[col];
        }
      }
    }
  }



//! @}
