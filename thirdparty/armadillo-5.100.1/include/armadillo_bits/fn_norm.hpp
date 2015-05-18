// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_norm
//! @{



template<typename T1>
arma_hot
inline
typename T1::pod_type
arma_vec_norm_1
  (
  const Proxy<T1>& P,
  const typename arma_not_cx<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::pod_type T;
  
  T acc = T(0);
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typename Proxy<T1>::ea_type A = P.get_ea();
    
    const uword N = P.get_n_elem();
    
    T acc1 = T(0);
    T acc2 = T(0);
    
    uword i,j;
    for(i=0, j=1; j<N; i+=2, j+=2)
      {
      acc1 += std::abs(A[i]);
      acc2 += std::abs(A[j]);
      }
    
    if(i < N)
      {
      acc1 += std::abs(A[i]);
      }
    
    acc = acc1 + acc2;
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    if(n_rows == 1)
      {
      for(uword col=0; col<n_cols; ++col)
        {
        acc += std::abs(P.at(0,col));
        }
      }
    else
      {
      T acc1 = T(0);
      T acc2 = T(0);
      
      for(uword col=0; col<n_cols; ++col)
        {
        uword i,j;
        
        for(i=0, j=1; j<n_rows; i+=2, j+=2)
          {
          acc1 += std::abs(P.at(i,col));
          acc2 += std::abs(P.at(j,col));
          }
        
        if(i < n_rows)
          {
          acc1 += std::abs(P.at(i,col));
          }
        }
      
      acc = acc1 + acc2;
      }
    }
  
  return acc;
  }



template<typename T1>
arma_hot
inline
typename T1::pod_type
arma_vec_norm_1
  (
  const Proxy<T1>& P,
  const typename arma_cx_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  T acc = T(0);
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typename Proxy<T1>::ea_type A = P.get_ea();
    
    const uword N = P.get_n_elem();
    
    for(uword i=0; i<N; ++i)
      {
      const std::complex<T>& X = A[i];
      
      const T a = X.real();
      const T b = X.imag();
      
      acc += std::sqrt( (a*a) + (b*b) );
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    if(n_rows == 1)
      {
      for(uword col=0; col<n_cols; ++col)
        {
        const std::complex<T>& X = P.at(0,col);
        
        const T a = X.real();
        const T b = X.imag();
        
        acc += std::sqrt( (a*a) + (b*b) );
        }
      }
    else
      {
      for(uword col=0; col<n_cols; ++col)
      for(uword row=0; row<n_rows; ++row)
        {
        const std::complex<T>& X = P.at(row,col);
        
        const T a = X.real();
        const T b = X.imag();
        
        acc += std::sqrt( (a*a) + (b*b) );
        }
      }
    }
  
  if( (acc != T(0)) && arma_isfinite(acc) )
    {
    return acc;
    }
  else
    {
    arma_extra_debug_print("arma_vec_norm_1(): detected possible underflow or overflow");
    
    const quasi_unwrap<typename Proxy<T1>::stored_type> R(P.Q);
    
    const uword N     = R.M.n_elem;
    const eT*   R_mem = R.M.memptr();
    
    T max_val = priv::most_neg<T>();
    
    for(uword i=0; i<N; ++i)
      {
      const std::complex<T>& X = R_mem[i];
      
      const T a = std::abs(X.real());
      const T b = std::abs(X.imag());
      
      if(a > max_val)  { max_val = a; }
      if(b > max_val)  { max_val = b; }
      }
    
    if(max_val == T(0))  { return T(0); }
    
    T alt_acc = T(0);
    
    for(uword i=0; i<N; ++i)
      {
      const std::complex<T>& X = R_mem[i];
      
      const T a = X.real() / max_val;
      const T b = X.imag() / max_val;
      
      alt_acc += std::sqrt( (a*a) + (b*b) );
      }
    
    return ( alt_acc * max_val );
    }
  }



template<typename eT>
arma_hot
inline
eT
arma_vec_norm_2_direct_mem_robust
  (
  const Mat<eT>& X
  )
  {
  arma_extra_debug_sigprint();
  
  const uword N = X.n_elem;
  const eT*   A = X.memptr();
  
  eT max_val = priv::most_neg<eT>();
  
  uword j;
  
  for(j=1; j<N; j+=2)
    {
    eT val_i = (*A);  A++;
    eT val_j = (*A);  A++;
    
    val_i = std::abs(val_i);
    val_j = std::abs(val_j);
    
    if(val_i > max_val)  { max_val = val_i; }
    if(val_j > max_val)  { max_val = val_j; }
    }
  
  if((j-1) < N)
    {
    const eT val_i = std::abs(*A);
    
    if(val_i > max_val)  { max_val = val_i; }
    }
  
  if(max_val == eT(0))  { return eT(0); }
  
  const eT* B = X.memptr();
  
  eT acc1 = eT(0);
  eT acc2 = eT(0);
  
  for(j=1; j<N; j+=2)
    {
    eT val_i = (*B);  B++;
    eT val_j = (*B);  B++;
    
    val_i /= max_val;
    val_j /= max_val;
    
    acc1 += val_i * val_i;
    acc2 += val_j * val_j;
    }
  
  if((j-1) < N)
    {
    const eT val_i = (*B) / max_val;
    
    acc1 += val_i * val_i;
    }
  
  return ( std::sqrt(acc1 + acc2) * max_val ); 
  }



template<typename eT>
arma_hot
inline
eT
arma_vec_norm_2_direct_mem_fast
  (
  const Mat<eT>& X
  )
  {
  arma_extra_debug_sigprint();
  
  const uword N = X.n_elem;
  const eT*   A = X.memptr();
  
  eT acc;
  
  #if defined(__FINITE_MATH_ONLY__) && (__FINITE_MATH_ONLY__ > 0)
    {
    eT acc1 = eT(0);
    
    if(memory::is_aligned(A))
      {
      memory::mark_as_aligned(A);
      
      for(uword i=0; i<N; ++i)  { const eT tmp_i = A[i];  acc1 += tmp_i * tmp_i; }
      }
    else
      {
      for(uword i=0; i<N; ++i)  { const eT tmp_i = A[i];  acc1 += tmp_i * tmp_i; }
      }
    
    acc = acc1;
    }
  #else
    {
    eT acc1 = eT(0);
    eT acc2 = eT(0);
    
    uword j;
    
    for(j=1; j<N; j+=2)
      {
      const eT tmp_i = (*A);  A++;
      const eT tmp_j = (*A);  A++;
      
      acc1 += tmp_i * tmp_i;
      acc2 += tmp_j * tmp_j;
      }
    
    if((j-1) < N)
      {
      const eT tmp_i = (*A);
      
      acc1 += tmp_i * tmp_i;
      }
    
    acc = acc1 + acc2;
    }
  #endif
  
  const eT sqrt_acc = std::sqrt(acc);
  
  if( (sqrt_acc != eT(0)) && arma_isfinite(sqrt_acc) )
    {
    return sqrt_acc;
    }
  else
    {
    arma_extra_debug_print("arma_vec_norm_2_direct_mem_fast(): detected possible underflow or overflow");
    
    return arma_vec_norm_2_direct_mem_robust(X);
    }
  }



template<typename T1>
arma_hot
inline
typename T1::pod_type
arma_vec_norm_2
  (
  const Proxy<T1>& P,
  const typename arma_not_cx<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const bool have_direct_mem = (is_Mat<typename Proxy<T1>::stored_type>::value) || (is_subview_col<typename Proxy<T1>::stored_type>::value);
  
  if(have_direct_mem)
    {
    const quasi_unwrap<typename Proxy<T1>::stored_type> tmp(P.Q);
    
    return arma_vec_norm_2_direct_mem_fast(tmp.M);
    }
  
  typedef typename T1::pod_type T;
  
  T acc = T(0);
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typename Proxy<T1>::ea_type A = P.get_ea();
    
    const uword N = P.get_n_elem();
    
    T acc1 = T(0);
    T acc2 = T(0);
    
    uword i,j;
    
    for(i=0, j=1; j<N; i+=2, j+=2)
      {
      const T tmp_i = A[i];
      const T tmp_j = A[j];
      
      acc1 += tmp_i * tmp_i;
      acc2 += tmp_j * tmp_j;
      }
    
    if(i < N)
      {
      const T tmp_i = A[i];
      
      acc1 += tmp_i * tmp_i;
      }
    
    acc = acc1 + acc2;
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    if(n_rows == 1)
      {
      for(uword col=0; col<n_cols; ++col)
        {
        const T tmp = P.at(0,col);
        
        acc += tmp * tmp;
        }
      }
    else
      {
      for(uword col=0; col<n_cols; ++col)
        {
        uword i,j;
        for(i=0, j=1; j<n_rows; i+=2, j+=2)
          {
          const T tmp_i = P.at(i,col);
          const T tmp_j = P.at(j,col);
          
          acc += tmp_i * tmp_i;
          acc += tmp_j * tmp_j;
          }
        
        if(i < n_rows)
          {
          const T tmp_i = P.at(i,col);
          
          acc += tmp_i * tmp_i;
          }
        }
      }
    }
  
  
  const T sqrt_acc = std::sqrt(acc);
  
  if( (sqrt_acc != T(0)) && arma_isfinite(sqrt_acc) )
    {
    return sqrt_acc;
    }
  else
    {
    arma_extra_debug_print("arma_vec_norm_2(): detected possible underflow or overflow");
  
    const quasi_unwrap<typename Proxy<T1>::stored_type> tmp(P.Q);
    
    return arma_vec_norm_2_direct_mem_robust(tmp.M);
    }
  }



template<typename T1>
arma_hot
inline
typename T1::pod_type
arma_vec_norm_2
  (
  const Proxy<T1>& P,
  const typename arma_cx_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  T acc = T(0);
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typename Proxy<T1>::ea_type A = P.get_ea();
    
    const uword N = P.get_n_elem();
    
    for(uword i=0; i<N; ++i)
      {
      const std::complex<T>& X = A[i];
      
      const T a = X.real();
      const T b = X.imag();
      
      acc += (a*a) + (b*b);
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    if(n_rows == 1)
      {
      for(uword col=0; col<n_cols; ++col)
        {
        const std::complex<T>& X = P.at(0,col);
        
        const T a = X.real();
        const T b = X.imag();
        
        acc += (a*a) + (b*b);
        }
      }
    else
      {
      for(uword col=0; col<n_cols; ++col)
      for(uword row=0; row<n_rows; ++row)
        {
        const std::complex<T>& X = P.at(row,col);
        
        const T a = X.real();
        const T b = X.imag();
        
        acc += (a*a) + (b*b);
        }
      }
    }
  
  const T sqrt_acc = std::sqrt(acc);
  
  if( (sqrt_acc != T(0)) && arma_isfinite(sqrt_acc) )
    {
    return sqrt_acc;
    }
  else
    {
    arma_extra_debug_print("arma_vec_norm_2(): detected possible underflow or overflow");
    
    const quasi_unwrap<typename Proxy<T1>::stored_type> R(P.Q);
    
    const uword N     = R.M.n_elem;
    const eT*   R_mem = R.M.memptr();
    
    T max_val = priv::most_neg<T>();
    
    for(uword i=0; i<N; ++i)
      {
      const T val_i = std::abs(R_mem[i]);
      
      if(val_i > max_val)  { max_val = val_i; }
      }
    
    if(max_val == T(0))  { return T(0); }
    
    T alt_acc = T(0);
    
    for(uword i=0; i<N; ++i)
      {
      const T val_i = std::abs(R_mem[i]) / max_val;
      
      alt_acc += val_i * val_i;
      }
    
    return ( std::sqrt(alt_acc) * max_val ); 
    }
  }



template<typename T1>
arma_hot
inline
typename T1::pod_type
arma_vec_norm_k
  (
  const Proxy<T1>& P,
  const int k
  )
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::pod_type T;
  
  T acc = T(0);
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typename Proxy<T1>::ea_type A = P.get_ea();
    
    const uword N = P.get_n_elem();
    
    uword i,j;
    
    for(i=0, j=1; j<N; i+=2, j+=2)
      {
      acc += std::pow(std::abs(A[i]), k);
      acc += std::pow(std::abs(A[j]), k);
      }
    
    if(i < N)
      {
      acc += std::pow(std::abs(A[i]), k);
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    if(n_rows != 1)
      {
      for(uword col=0; col < n_cols; ++col)
      for(uword row=0; row < n_rows; ++row)
        {
        acc += std::pow(std::abs(P.at(row,col)), k);
        }
      }
    else
      {
      for(uword col=0; col < n_cols; ++col)
        {
        acc += std::pow(std::abs(P.at(0,col)), k);
        }
      }
    }
  
  return std::pow(acc, T(1)/T(k));
  }



template<typename T1>
arma_hot
inline
typename T1::pod_type
arma_vec_norm_max(const Proxy<T1>& P)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::pod_type T;
  
  const uword N = P.get_n_elem();
  
  T max_val = (N != 1) ? priv::most_neg<T>() : std::abs(P[0]);
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typename Proxy<T1>::ea_type A = P.get_ea();
    
    uword i,j;
    for(i=0, j=1; j<N; i+=2, j+=2)
      {
      const T tmp_i = std::abs(A[i]);
      const T tmp_j = std::abs(A[j]);
      
      if(max_val < tmp_i) { max_val = tmp_i; }
      if(max_val < tmp_j) { max_val = tmp_j; }
      }
    
    if(i < N)
      {
      const T tmp_i = std::abs(A[i]);
      
      if(max_val < tmp_i) { max_val = tmp_i; }
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    if(n_rows != 1)
      {
      for(uword col=0; col < n_cols; ++col)
      for(uword row=0; row < n_rows; ++row)
        {
        const T tmp = std::abs(P.at(row,col));
        
        if(max_val < tmp) { max_val = tmp; }
        }
      }
    else
      {
      for(uword col=0; col < n_cols; ++col)
        {
        const T tmp = std::abs(P.at(0,col));
        
        if(max_val < tmp) { max_val = tmp; }
        }
      }
    }
  
  return max_val;
  }



template<typename T1>
arma_hot
inline
typename T1::pod_type
arma_vec_norm_min(const Proxy<T1>& P)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::pod_type T;
  
  const uword N = P.get_n_elem();
  
  T min_val = (N != 1) ? priv::most_pos<T>() : std::abs(P[0]);
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    typename Proxy<T1>::ea_type A = P.get_ea();
    
    uword i,j;
    for(i=0, j=1; j<N; i+=2, j+=2)
      {
      const T tmp_i = std::abs(A[i]);
      const T tmp_j = std::abs(A[j]);
      
      if(min_val > tmp_i) { min_val = tmp_i; }
      if(min_val > tmp_j) { min_val = tmp_j; }
      }
    
    if(i < N)
      {
      const T tmp_i = std::abs(A[i]);
      
      if(min_val > tmp_i) { min_val = tmp_i; }
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    if(n_rows != 1)
      {
      for(uword col=0; col < n_cols; ++col)
      for(uword row=0; row < n_rows; ++row)
        {
        const T tmp = std::abs(P.at(row,col));
        
        if(min_val > tmp) { min_val = tmp; }
        }
      }
    else
      {
      for(uword col=0; col < n_cols; ++col)
        {
        const T tmp = std::abs(P.at(0,col));
        
        if(min_val > tmp) { min_val = tmp; }
        }
      }
    }
  
  return min_val;
  }



template<typename T1>
inline
typename T1::pod_type
arma_mat_norm_1(const Proxy<T1>& P)
  {
  arma_extra_debug_sigprint();
  
  // TODO: this can be sped up with a dedicated implementation
  return as_scalar( max( sum(abs(P.Q), 0), 1) );
  }



template<typename T1>
inline
typename T1::pod_type
arma_mat_norm_2(const Proxy<T1>& P)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::pod_type   T;
  
  // TODO: is the SVD based approach only valid for square matrices?
  
  Col<T> S;
  svd(S, P.Q);
  
  return (S.n_elem > 0) ? max(S) : T(0);
  }



template<typename T1>
inline
typename T1::pod_type
arma_mat_norm_inf(const Proxy<T1>& P)
  {
  arma_extra_debug_sigprint();
  
  // TODO: this can be sped up with a dedicated implementation
  return as_scalar( max( sum(abs(P.Q), 1), 0) );
  }



template<typename T1>
inline
arma_warn_unused
typename enable_if2< is_arma_type<T1>::value, typename T1::pod_type >::result
norm
  (
  const T1& X,
  const uword k = uword(2),
  const typename arma_real_or_cx_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::pod_type T;
  
  const Proxy<T1> P(X);
  
  if(P.get_n_elem() == 0)
    {
    return T(0);
    }
  
  const bool is_vec = (P.get_n_rows() == 1) || (P.get_n_cols() == 1);
  
  if(is_vec)
    {
    switch(k)
      {
      case 1:
        return arma_vec_norm_1(P);
        break;
      
      case 2:
        return arma_vec_norm_2(P);
        break;
      
      default:
        {
        arma_debug_check( (k == 0), "norm(): k must be greater than zero"   );
        return arma_vec_norm_k(P, int(k));
        }
      }
    }
  else
    {
    switch(k)
      {
      case 1:
        return arma_mat_norm_1(P);
        break;
      
      case 2:
        return arma_mat_norm_2(P);
        break;
      
      default:
        arma_stop("norm(): unsupported matrix norm type");
        return T(0);
      }
    }
  
  return T(0);  // prevent erroneous compiler warnings
  }



template<typename T1>
inline
arma_warn_unused
typename enable_if2< is_arma_type<T1>::value, typename T1::pod_type >::result
norm
  (
  const T1& X,
  const char* method,
  const typename arma_real_or_cx_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::pod_type T;
  
  const Proxy<T1> P(X);
  
  if(P.get_n_elem() == 0)
    {
    return T(0);
    }
  
  const char sig    = (method != NULL) ? method[0] : char(0);
  const bool is_vec = (P.get_n_rows() == 1) || (P.get_n_cols() == 1);
  
  if(is_vec)
    {
    if( (sig == 'i') || (sig == 'I') || (sig == '+') )   // max norm
      {
      return arma_vec_norm_max(P);
      }
    else
    if(sig == '-')   // min norm
      {
      return arma_vec_norm_min(P);
      }
    else
    if( (sig == 'f') || (sig == 'F') )
      {
      return arma_vec_norm_2(P);
      }
    else
      {
      arma_stop("norm(): unsupported vector norm type");
      return T(0);
      }
    }
  else
    {
    if( (sig == 'i') || (sig == 'I') || (sig == '+') )   // inf norm
      {
      return arma_mat_norm_inf(P);
      }
    else
    if( (sig == 'f') || (sig == 'F') )
      {
      return arma_vec_norm_2(P);
      }
    else
      {
      arma_stop("norm(): unsupported matrix norm type");
      return T(0);
      }
    }
  }



//! @}
