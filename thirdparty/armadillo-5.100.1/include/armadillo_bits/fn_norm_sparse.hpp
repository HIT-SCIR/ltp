// Copyright (C) 2012-2014 Conrad Sanderson
// Copyright (C) 2012-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_norm
//! @{


//
// norms for sparse matrices


template<typename T1>
inline
typename T1::pod_type
arma_mat_norm_1(const SpProxy<T1>& P)
  {
  arma_extra_debug_sigprint();
  
  // TODO: this can be sped up with a dedicated implementation
  return as_scalar( max( sum(abs(P.Q), 0), 1) );
  }



template<typename T1>
inline
typename T1::pod_type
arma_mat_norm_2(const SpProxy<T1>& P, const typename arma_real_only<typename T1::elem_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  // norm = sqrt( largest eigenvalue of (A^H)*A ), where ^H is the conjugate transpose
  // http://math.stackexchange.com/questions/4368/computing-the-largest-eigenvalue-of-a-very-large-sparse-matrix
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  const unwrap_spmat<typename SpProxy<T1>::stored_type> tmp(P.Q);
  
  const SpMat<eT>& A = tmp.M;
  const SpMat<eT>  B = trans(A);
  
  const SpMat<eT>  C = (A.n_rows <= A.n_cols) ? (A*B) : (B*A);
  
  const Col<T> eigval = eigs_sym(C, 1);
  
  return (eigval.n_elem > 0) ? std::sqrt(eigval[0]) : T(0);
  }



template<typename T1>
inline
typename T1::pod_type
arma_mat_norm_2(const SpProxy<T1>& P, const typename arma_cx_only<typename T1::elem_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  //typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  arma_ignore(P);
  arma_stop("norm(): unimplemented norm type for complex sparse matrices");
  
  return T(0);
  
  // const unwrap_spmat<typename SpProxy<T1>::stored_type> tmp(P.Q);
  // 
  // const SpMat<eT>& A = tmp.M;
  // const SpMat<eT>  B = trans(A);
  // 
  // const SpMat<eT>  C = (A.n_rows <= A.n_cols) ? (A*B) : (B*A);
  // 
  // const Col<eT> eigval = eigs_gen(C, 1);
  }



template<typename T1>
inline
typename T1::pod_type
arma_mat_norm_inf(const SpProxy<T1>& P)
  {
  arma_extra_debug_sigprint();
  
  // TODO: this can be sped up with a dedicated implementation
  return as_scalar( max( sum(abs(P.Q), 1), 0) );
  }



template<typename T1>
inline
arma_warn_unused
typename enable_if2< is_arma_sparse_type<T1>::value, typename T1::pod_type >::result
norm
  (
  const T1& X,
  const uword k = uword(2),
  const typename arma_real_or_cx_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  const SpProxy<T1> P(X);
  
  if(P.get_n_nonzero() == 0)
    {
    return T(0);
    }
  
  const bool is_vec = (P.get_n_rows() == 1) || (P.get_n_cols() == 1);
  
  if(is_vec == true)
    {
    const unwrap_spmat<typename SpProxy<T1>::stored_type> tmp(P.Q);
    const SpMat<eT>& A = tmp.M;
    
    // create a fake dense vector to allow reuse of code for dense vectors
    Col<eT> fake_vector( access::rwp(A.values), A.n_nonzero, false );
    
    const Proxy< Col<eT> > P_fake_vector(fake_vector);
    
    switch(k)
      {
      case 1:
        return arma_vec_norm_1(P_fake_vector);
        break;
      
      case 2:
        return arma_vec_norm_2(P_fake_vector);
        break;
      
      default:
        {
        arma_debug_check( (k == 0), "norm(): k must be greater than zero"   );
        return arma_vec_norm_k(P_fake_vector, int(k));
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
        arma_stop("norm(): unsupported or unimplemented norm type for sparse matrices");
        return T(0);
      }
    }
  }



template<typename T1>
inline
arma_warn_unused
typename enable_if2< is_arma_sparse_type<T1>::value, typename T1::pod_type >::result
norm
  (
  const T1& X,
  const char* method,
  const typename arma_real_or_cx_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  const SpProxy<T1> P(X);
  
  if(P.get_n_nonzero() == 0)
    {
    return T(0);
    }
  
  
  const unwrap_spmat<typename SpProxy<T1>::stored_type> tmp(P.Q);
  const SpMat<eT>& A = tmp.M;
  
  // create a fake dense vector to allow reuse of code for dense vectors
  Col<eT> fake_vector( access::rwp(A.values), A.n_nonzero, false );
  
  const Proxy< Col<eT> > P_fake_vector(fake_vector);
  
  
  const char sig    = (method != NULL) ? method[0] : char(0);
  const bool is_vec = (P.get_n_rows() == 1) || (P.get_n_cols() == 1);
  
  if(is_vec == true)
    {
    if( (sig == 'i') || (sig == 'I') || (sig == '+') )   // max norm
      {
      return arma_vec_norm_max(P_fake_vector);
      }
    else
    if(sig == '-')   // min norm
      {
      const T val = arma_vec_norm_min(P_fake_vector);
      
      if( P.get_n_nonzero() < P.get_n_elem() )
        {
        return (std::min)(T(0), val);
        }
      else
        {
        return val;
        }
      }
    else
    if( (sig == 'f') || (sig == 'F') )
      {
      return arma_vec_norm_2(P_fake_vector);
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
      return arma_vec_norm_2(P_fake_vector);
      }
    else
      {
      arma_stop("norm(): unsupported matrix norm type");
      return T(0);
      }
    }
  }



//! @}
