// Copyright (C) 2008-2015 Conrad Sanderson
// Copyright (C) 2008-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_misc
//! @{



//! \brief
//! Generate a vector with 'num' elements.
//! The values of the elements linearly increase from 'start' upto (and including) 'end'.

template<typename vec_type>
inline
typename
enable_if2
  <
  is_Mat<vec_type>::value,
  vec_type
  >::result
linspace
  (
  const typename vec_type::pod_type start,
  const typename vec_type::pod_type end,
  const uword num = 100u
  )
  {
  arma_extra_debug_sigprint();
  
  typedef typename vec_type::elem_type eT;
  typedef typename vec_type::pod_type   T;
  
  vec_type x;
    
  if(num >= 2)
    {
    x.set_size(num);
    
    eT* x_mem = x.memptr();
    
    const uword num_m1 = num - 1;
    
    if(is_non_integral<T>::value == true)
      {
      const T delta = (end-start)/T(num_m1);
      
      for(uword i=0; i<num_m1; ++i)
        {
        x_mem[i] = eT(start + i*delta);
        }
      
      x_mem[num_m1] = eT(end);
      }
    else
      {
      const double delta = (end >= start) ? double(end-start)/double(num_m1) : -double(start-end)/double(num_m1);
      
      for(uword i=0; i<num_m1; ++i)
        {
        x_mem[i] = eT(double(start) + i*delta);
        }
      
      x_mem[num_m1] = eT(end);
      }
    
    return x;
    }
  else
    {
    x.set_size(1);
    
    x[0] = eT(end);
    }
  
  return x;
  }



inline
mat
linspace(const double start, const double end, const uword num = 100u)
  {
  arma_extra_debug_sigprint();
  return linspace<mat>(start, end, num);
  }



//
// log_exp_add

template<typename eT>
inline
typename arma_real_only<eT>::result
log_add_exp(eT log_a, eT log_b)
  {
  if(log_a < log_b)
    {
    std::swap(log_a, log_b);
    }
  
  const eT negdelta = log_b - log_a;
  
  if( (negdelta < Datum<eT>::log_min) || (arma_isfinite(negdelta) == false) )
    {
    return log_a;
    }
  else
    {
    return (log_a + arma_log1p(std::exp(negdelta)));
    }
  }



// for compatibility with earlier versions
template<typename eT>
inline
typename arma_real_only<eT>::result
log_add(eT log_a, eT log_b)
  {
  return log_add_exp(log_a, log_b);
  }
  


template<typename eT>
arma_inline
arma_warn_unused
bool
is_finite(const eT x, const typename arma_scalar_only<eT>::result* junk = 0)
  {
  arma_ignore(junk);
  
  return arma_isfinite(x);
  }



template<typename T1>
inline
arma_warn_unused
typename
enable_if2
  <
  is_arma_type<T1>::value,
  bool
  >::result
is_finite(const T1& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const Proxy<T1> P(X);
  
  const bool have_direct_mem = (is_Mat<typename Proxy<T1>::stored_type>::value) || (is_subview_col<typename Proxy<T1>::stored_type>::value);
  
  if(have_direct_mem)
    {
    const quasi_unwrap<typename Proxy<T1>::stored_type> tmp(P.Q);
    
    return tmp.M.is_finite();
    }
  
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    const typename Proxy<T1>::ea_type Pea = P.get_ea();
    
    const uword n_elem = P.get_n_elem();
    
    uword i,j;
    
    for(i=0, j=1; j<n_elem; i+=2, j+=2)
      {
      const eT val_i = Pea[i];
      const eT val_j = Pea[j];
      
      if( (arma_isfinite(val_i) == false) || (arma_isfinite(val_j) == false) )  { return false; }
      }
    
    if(i < n_elem)
      {
      if(arma_isfinite(Pea[i]) == false)  { return false; }
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    for(uword col=0; col<n_cols; ++col)
    for(uword row=0; row<n_rows; ++row)
      {
      if(arma_isfinite(P.at(row,col)) == false)  { return false; }
      }
    }
  
  return true;
  }



template<typename T1>
inline
arma_warn_unused
bool
is_finite(const SpBase<typename T1::elem_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  const SpProxy<T1> P(X.get_ref());
  
  if(is_SpMat<typename SpProxy<T1>::stored_type>::value)
    {
    const unwrap_spmat<typename SpProxy<T1>::stored_type> tmp(P.Q);
    
    return tmp.M.is_finite();
    }
  else
    {
    typename SpProxy<T1>::const_iterator_type it     = P.begin();
    typename SpProxy<T1>::const_iterator_type it_end = P.end();
    
    while(it != it_end)
      {
      if(arma_isfinite(*it) == false)  { return false; }
      ++it;
      }
    }
  
  return true;
  }



template<typename T1>
inline
arma_warn_unused
bool
is_finite(const BaseCube<typename T1::elem_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap_cube<T1> tmp(X.get_ref());
  const Cube<eT>& A =   tmp.M;
  
  return A.is_finite();
  }



//! DO NOT USE IN NEW CODE; change instances of inv(sympd(X)) to inv_sympd(X)
template<typename T1>
arma_deprecated
inline
const T1&
sympd(const Base<typename T1::elem_type,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  return X.get_ref();
  }



template<typename eT>
inline
void
swap(Mat<eT>& A, Mat<eT>& B)
  {
  arma_extra_debug_sigprint();
  
  A.swap(B);
  }



template<typename eT>
inline
void
swap(Cube<eT>& A, Cube<eT>& B)
  {
  arma_extra_debug_sigprint();
  
  A.swap(B);
  }



//! @}
