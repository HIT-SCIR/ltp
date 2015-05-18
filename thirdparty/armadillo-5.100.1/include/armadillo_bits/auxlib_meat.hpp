// Copyright (C) 2008-2015 Conrad Sanderson
// Copyright (C) 2008-2015 NICTA (www.nicta.com.au)
// Copyright (C) 2009 Edmund Highcock
// Copyright (C) 2011 James Sanders
// Copyright (C) 2011 Stanislav Funiak
// Copyright (C) 2012 Eric Jon Sundstrom
// Copyright (C) 2012 Michael McNeil Forbes
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup auxlib
//! @{



//! immediate matrix inverse
template<typename eT, typename T1>
inline
bool
auxlib::inv(Mat<eT>& out, const Base<eT,T1>& X, const bool slow)
  {
  arma_extra_debug_sigprint();
  
  out = X.get_ref();
  
  arma_debug_check( (out.is_square() == false), "inv(): given matrix is not square" );
  
  bool status = false;
  
  const uword N = out.n_rows;
  
  if( (N <= 4) && (slow == false) )
    {
    Mat<eT> tmp(N,N);
    
    status = auxlib::inv_noalias_tinymat(tmp, out, N);
    
    if(status == true)
      {
      arrayops::copy( out.memptr(), tmp.memptr(), tmp.n_elem );
      }
    }
  
  if( (N > 4) || (status == false) )
    {
    status = auxlib::inv_inplace_lapack(out);
    }
  
  return status;
  }



template<typename eT>
inline
bool
auxlib::inv(Mat<eT>& out, const Mat<eT>& X, const bool slow)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (X.is_square() == false), "inv(): given matrix is not square" );
  
  bool status = false;
  
  const uword N = X.n_rows;
  
  if( (N <= 4) && (slow == false) )
    {
    if(&out != &X)
      {
      out.set_size(N,N);
      
      status = auxlib::inv_noalias_tinymat(out, X, N);
      }
    else
      {
      Mat<eT> tmp(N,N);
      
      status = auxlib::inv_noalias_tinymat(tmp, X, N);
      
      if(status == true)
        {
        arrayops::copy( out.memptr(), tmp.memptr(), tmp.n_elem );
        }
      }
    }
  
  if( (N > 4) || (status == false) )
    {
    out = X;
    status = auxlib::inv_inplace_lapack(out);
    }
  
  return status;
  }



template<typename eT>
inline
bool
auxlib::inv_noalias_tinymat(Mat<eT>& out, const Mat<eT>& X, const uword N)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  const T det_min = (is_float<T>::value) ? T(1e-19) : T(1e-154);
  
  bool calc_ok = true;
  
  const eT* Xm   =   X.memptr();
        eT* outm = out.memptr();  // NOTE: the output matrix is assumed to have the correct size
        
  switch(N)
    {
    case 1:
      {
      outm[0] = eT(1) / Xm[0];
      };
      break;
      
    case 2:
      {
      const eT a = Xm[pos<0,0>::n2];
      const eT b = Xm[pos<0,1>::n2];
      const eT c = Xm[pos<1,0>::n2];
      const eT d = Xm[pos<1,1>::n2];
      
      const eT det_val = (a*d - b*c);
      
      if(std::abs(det_val) >= det_min)
        {
        outm[pos<0,0>::n2] =  d / det_val;
        outm[pos<0,1>::n2] = -b / det_val;
        outm[pos<1,0>::n2] = -c / det_val;
        outm[pos<1,1>::n2] =  a / det_val;
        }
      else
        {
        calc_ok = false;
        }
      };
      break;
    
    case 3:
      {
      const eT det_val = auxlib::det_tinymat(X,3);
      
      if(std::abs(det_val) >= det_min)
        {
        outm[pos<0,0>::n3] =  (Xm[pos<2,2>::n3]*Xm[pos<1,1>::n3] - Xm[pos<2,1>::n3]*Xm[pos<1,2>::n3]) / det_val;
        outm[pos<1,0>::n3] = -(Xm[pos<2,2>::n3]*Xm[pos<1,0>::n3] - Xm[pos<2,0>::n3]*Xm[pos<1,2>::n3]) / det_val;
        outm[pos<2,0>::n3] =  (Xm[pos<2,1>::n3]*Xm[pos<1,0>::n3] - Xm[pos<2,0>::n3]*Xm[pos<1,1>::n3]) / det_val;
        
        outm[pos<0,1>::n3] = -(Xm[pos<2,2>::n3]*Xm[pos<0,1>::n3] - Xm[pos<2,1>::n3]*Xm[pos<0,2>::n3]) / det_val;
        outm[pos<1,1>::n3] =  (Xm[pos<2,2>::n3]*Xm[pos<0,0>::n3] - Xm[pos<2,0>::n3]*Xm[pos<0,2>::n3]) / det_val;
        outm[pos<2,1>::n3] = -(Xm[pos<2,1>::n3]*Xm[pos<0,0>::n3] - Xm[pos<2,0>::n3]*Xm[pos<0,1>::n3]) / det_val;
        
        outm[pos<0,2>::n3] =  (Xm[pos<1,2>::n3]*Xm[pos<0,1>::n3] - Xm[pos<1,1>::n3]*Xm[pos<0,2>::n3]) / det_val;
        outm[pos<1,2>::n3] = -(Xm[pos<1,2>::n3]*Xm[pos<0,0>::n3] - Xm[pos<1,0>::n3]*Xm[pos<0,2>::n3]) / det_val;
        outm[pos<2,2>::n3] =  (Xm[pos<1,1>::n3]*Xm[pos<0,0>::n3] - Xm[pos<1,0>::n3]*Xm[pos<0,1>::n3]) / det_val;
        
        const eT check_val = Xm[pos<0,0>::n3]*outm[pos<0,0>::n3] + Xm[pos<0,1>::n3]*outm[pos<1,0>::n3] + Xm[pos<0,2>::n3]*outm[pos<2,0>::n3];
        
        const  T max_diff  = (is_float<T>::value) ? T(1e-4) : T(1e-10);
        
        if(std::abs(T(1) - check_val) > max_diff)
          {
          calc_ok = false;
          }
        }
      else
        {
        calc_ok = false;
        }
      };
      break;
    
    case 4:
      {
      const eT det_val = auxlib::det_tinymat(X,4);
      
      if(std::abs(det_val) >= det_min)
        {
        outm[pos<0,0>::n4] = ( Xm[pos<1,2>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,1>::n4] - Xm[pos<1,3>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,1>::n4] + Xm[pos<1,3>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,2>::n4] - Xm[pos<1,1>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,2>::n4] - Xm[pos<1,2>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,3>::n4] + Xm[pos<1,1>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,3>::n4] ) / det_val;
        outm[pos<1,0>::n4] = ( Xm[pos<1,3>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,0>::n4] - Xm[pos<1,2>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,0>::n4] - Xm[pos<1,3>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,2>::n4] + Xm[pos<1,0>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,2>::n4] + Xm[pos<1,2>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,3>::n4] - Xm[pos<1,0>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,3>::n4] ) / det_val;
        outm[pos<2,0>::n4] = ( Xm[pos<1,1>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,0>::n4] - Xm[pos<1,3>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,0>::n4] + Xm[pos<1,3>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,1>::n4] - Xm[pos<1,0>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,1>::n4] - Xm[pos<1,1>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,3>::n4] + Xm[pos<1,0>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,3>::n4] ) / det_val;
        outm[pos<3,0>::n4] = ( Xm[pos<1,2>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,0>::n4] - Xm[pos<1,1>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,0>::n4] - Xm[pos<1,2>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,1>::n4] + Xm[pos<1,0>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,1>::n4] + Xm[pos<1,1>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,2>::n4] - Xm[pos<1,0>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,2>::n4] ) / det_val;
        
        outm[pos<0,1>::n4] = ( Xm[pos<0,3>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,1>::n4] - Xm[pos<0,2>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,1>::n4] - Xm[pos<0,3>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,2>::n4] + Xm[pos<0,1>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,2>::n4] + Xm[pos<0,2>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,3>::n4] - Xm[pos<0,1>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,3>::n4] ) / det_val;
        outm[pos<1,1>::n4] = ( Xm[pos<0,2>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,0>::n4] - Xm[pos<0,3>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,0>::n4] + Xm[pos<0,3>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,2>::n4] - Xm[pos<0,0>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,2>::n4] - Xm[pos<0,2>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,3>::n4] + Xm[pos<0,0>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,3>::n4] ) / det_val;
        outm[pos<2,1>::n4] = ( Xm[pos<0,3>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,0>::n4] - Xm[pos<0,1>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,0>::n4] - Xm[pos<0,3>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,1>::n4] + Xm[pos<0,0>::n4]*Xm[pos<2,3>::n4]*Xm[pos<3,1>::n4] + Xm[pos<0,1>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,3>::n4] - Xm[pos<0,0>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,3>::n4] ) / det_val;
        outm[pos<3,1>::n4] = ( Xm[pos<0,1>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,0>::n4] - Xm[pos<0,2>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,0>::n4] + Xm[pos<0,2>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,1>::n4] - Xm[pos<0,0>::n4]*Xm[pos<2,2>::n4]*Xm[pos<3,1>::n4] - Xm[pos<0,1>::n4]*Xm[pos<2,0>::n4]*Xm[pos<3,2>::n4] + Xm[pos<0,0>::n4]*Xm[pos<2,1>::n4]*Xm[pos<3,2>::n4] ) / det_val;
        
        outm[pos<0,2>::n4] = ( Xm[pos<0,2>::n4]*Xm[pos<1,3>::n4]*Xm[pos<3,1>::n4] - Xm[pos<0,3>::n4]*Xm[pos<1,2>::n4]*Xm[pos<3,1>::n4] + Xm[pos<0,3>::n4]*Xm[pos<1,1>::n4]*Xm[pos<3,2>::n4] - Xm[pos<0,1>::n4]*Xm[pos<1,3>::n4]*Xm[pos<3,2>::n4] - Xm[pos<0,2>::n4]*Xm[pos<1,1>::n4]*Xm[pos<3,3>::n4] + Xm[pos<0,1>::n4]*Xm[pos<1,2>::n4]*Xm[pos<3,3>::n4] ) / det_val;
        outm[pos<1,2>::n4] = ( Xm[pos<0,3>::n4]*Xm[pos<1,2>::n4]*Xm[pos<3,0>::n4] - Xm[pos<0,2>::n4]*Xm[pos<1,3>::n4]*Xm[pos<3,0>::n4] - Xm[pos<0,3>::n4]*Xm[pos<1,0>::n4]*Xm[pos<3,2>::n4] + Xm[pos<0,0>::n4]*Xm[pos<1,3>::n4]*Xm[pos<3,2>::n4] + Xm[pos<0,2>::n4]*Xm[pos<1,0>::n4]*Xm[pos<3,3>::n4] - Xm[pos<0,0>::n4]*Xm[pos<1,2>::n4]*Xm[pos<3,3>::n4] ) / det_val;
        outm[pos<2,2>::n4] = ( Xm[pos<0,1>::n4]*Xm[pos<1,3>::n4]*Xm[pos<3,0>::n4] - Xm[pos<0,3>::n4]*Xm[pos<1,1>::n4]*Xm[pos<3,0>::n4] + Xm[pos<0,3>::n4]*Xm[pos<1,0>::n4]*Xm[pos<3,1>::n4] - Xm[pos<0,0>::n4]*Xm[pos<1,3>::n4]*Xm[pos<3,1>::n4] - Xm[pos<0,1>::n4]*Xm[pos<1,0>::n4]*Xm[pos<3,3>::n4] + Xm[pos<0,0>::n4]*Xm[pos<1,1>::n4]*Xm[pos<3,3>::n4] ) / det_val;
        outm[pos<3,2>::n4] = ( Xm[pos<0,2>::n4]*Xm[pos<1,1>::n4]*Xm[pos<3,0>::n4] - Xm[pos<0,1>::n4]*Xm[pos<1,2>::n4]*Xm[pos<3,0>::n4] - Xm[pos<0,2>::n4]*Xm[pos<1,0>::n4]*Xm[pos<3,1>::n4] + Xm[pos<0,0>::n4]*Xm[pos<1,2>::n4]*Xm[pos<3,1>::n4] + Xm[pos<0,1>::n4]*Xm[pos<1,0>::n4]*Xm[pos<3,2>::n4] - Xm[pos<0,0>::n4]*Xm[pos<1,1>::n4]*Xm[pos<3,2>::n4] ) / det_val;
        
        outm[pos<0,3>::n4] = ( Xm[pos<0,3>::n4]*Xm[pos<1,2>::n4]*Xm[pos<2,1>::n4] - Xm[pos<0,2>::n4]*Xm[pos<1,3>::n4]*Xm[pos<2,1>::n4] - Xm[pos<0,3>::n4]*Xm[pos<1,1>::n4]*Xm[pos<2,2>::n4] + Xm[pos<0,1>::n4]*Xm[pos<1,3>::n4]*Xm[pos<2,2>::n4] + Xm[pos<0,2>::n4]*Xm[pos<1,1>::n4]*Xm[pos<2,3>::n4] - Xm[pos<0,1>::n4]*Xm[pos<1,2>::n4]*Xm[pos<2,3>::n4] ) / det_val;
        outm[pos<1,3>::n4] = ( Xm[pos<0,2>::n4]*Xm[pos<1,3>::n4]*Xm[pos<2,0>::n4] - Xm[pos<0,3>::n4]*Xm[pos<1,2>::n4]*Xm[pos<2,0>::n4] + Xm[pos<0,3>::n4]*Xm[pos<1,0>::n4]*Xm[pos<2,2>::n4] - Xm[pos<0,0>::n4]*Xm[pos<1,3>::n4]*Xm[pos<2,2>::n4] - Xm[pos<0,2>::n4]*Xm[pos<1,0>::n4]*Xm[pos<2,3>::n4] + Xm[pos<0,0>::n4]*Xm[pos<1,2>::n4]*Xm[pos<2,3>::n4] ) / det_val;
        outm[pos<2,3>::n4] = ( Xm[pos<0,3>::n4]*Xm[pos<1,1>::n4]*Xm[pos<2,0>::n4] - Xm[pos<0,1>::n4]*Xm[pos<1,3>::n4]*Xm[pos<2,0>::n4] - Xm[pos<0,3>::n4]*Xm[pos<1,0>::n4]*Xm[pos<2,1>::n4] + Xm[pos<0,0>::n4]*Xm[pos<1,3>::n4]*Xm[pos<2,1>::n4] + Xm[pos<0,1>::n4]*Xm[pos<1,0>::n4]*Xm[pos<2,3>::n4] - Xm[pos<0,0>::n4]*Xm[pos<1,1>::n4]*Xm[pos<2,3>::n4] ) / det_val;
        outm[pos<3,3>::n4] = ( Xm[pos<0,1>::n4]*Xm[pos<1,2>::n4]*Xm[pos<2,0>::n4] - Xm[pos<0,2>::n4]*Xm[pos<1,1>::n4]*Xm[pos<2,0>::n4] + Xm[pos<0,2>::n4]*Xm[pos<1,0>::n4]*Xm[pos<2,1>::n4] - Xm[pos<0,0>::n4]*Xm[pos<1,2>::n4]*Xm[pos<2,1>::n4] - Xm[pos<0,1>::n4]*Xm[pos<1,0>::n4]*Xm[pos<2,2>::n4] + Xm[pos<0,0>::n4]*Xm[pos<1,1>::n4]*Xm[pos<2,2>::n4] ) / det_val;
        
        const eT check_val = Xm[pos<0,0>::n4]*outm[pos<0,0>::n4] + Xm[pos<0,1>::n4]*outm[pos<1,0>::n4] + Xm[pos<0,2>::n4]*outm[pos<2,0>::n4] + Xm[pos<0,3>::n4]*outm[pos<3,0>::n4];
        
        const  T max_diff  = (is_float<T>::value) ? T(1e-4) : T(1e-10);
        
        if(std::abs(T(1) - check_val) > max_diff)
          {
          calc_ok = false;
          }
        }
      else
        {
        calc_ok = false;
        }
      };
      break;
    
    default:
      ;
    }
  
  return calc_ok;
  }



template<typename eT>
inline
bool
auxlib::inv_inplace_lapack(Mat<eT>& out)
  {
  arma_extra_debug_sigprint();
  
  if(out.is_empty())
    {
    return true;
    }
  
  #if defined(ARMA_USE_ATLAS)
    {
    arma_debug_assert_atlas_size(out);
    
    podarray<int> ipiv(out.n_rows);
    
    int info = atlas::clapack_getrf(atlas::CblasColMajor, out.n_rows, out.n_cols, out.memptr(), out.n_rows, ipiv.memptr());
    
    if(info == 0)
      {
      info = atlas::clapack_getri(atlas::CblasColMajor, out.n_rows, out.memptr(), out.n_rows, ipiv.memptr());
      }
    
    return (info == 0);
    }
  #elif defined(ARMA_USE_LAPACK)
    {
    arma_debug_assert_blas_size(out);
    
    blas_int n_rows = out.n_rows;
    blas_int lwork  = (std::max)(blas_int(podarray_prealloc_n_elem::val), n_rows);
    blas_int info   = 0;
    
    podarray<blas_int> ipiv(out.n_rows);
    
    if(n_rows > 16)
      {
      eT        work_query[2];
      blas_int lwork_query = -1;
      
      lapack::getri(&n_rows, out.memptr(), &n_rows, ipiv.memptr(), &work_query[0], &lwork_query, &info);
      
      if(info == 0)
        {
        const blas_int lwork_proposed = static_cast<blas_int>( access::tmp_real(work_query[0]) );
        
        if(lwork_proposed > lwork)
          {
          lwork = lwork_proposed;
          }
        }
      else
        {
        return false;
        }
      }
    
    podarray<eT> work( static_cast<uword>(lwork) );
    
    lapack::getrf(&n_rows, &n_rows, out.memptr(), &n_rows, ipiv.memptr(), &info);
    
    if(info == 0)
      {
      lapack::getri(&n_rows, out.memptr(), &n_rows, ipiv.memptr(), work.memptr(), &lwork, &info);
      }
    
    return (info == 0);
    }
  #else
    {
    arma_stop("inv(): use of ATLAS or LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename eT, typename T1>
inline
bool
auxlib::inv_tr(Mat<eT>& out, const Base<eT,T1>& X, const uword layout)
  {
  arma_extra_debug_sigprint();
  
  out = X.get_ref();
  
  arma_debug_check( (out.is_square() == false), "inv(): given matrix is not square" );
  
  if(out.is_empty())
    {
    return true;
    }
  
  bool status;
  
  #if defined(ARMA_USE_LAPACK)
    {
    arma_debug_assert_blas_size(out);
    
    char     uplo = (layout == 0) ? 'U' : 'L';
    char     diag = 'N';
    blas_int n    = blas_int(out.n_rows);
    blas_int info = 0;
    
    lapack::trtri(&uplo, &diag, &n, out.memptr(), &n, &info);
    
    status = (info == 0);
    }
  #else
    {
    arma_ignore(layout);
    arma_stop("inv(): use of LAPACK needs to be enabled");
    status = false;
    }
  #endif
  
  
  if(status == true)
    {
    if(layout == 0)
      {
      // upper triangular
      out = trimatu(out);
      }
    else
      {
      // lower triangular      
      out = trimatl(out);
      }
    }
  
  return status;
  }



template<typename eT, typename T1>
inline
bool
auxlib::inv_sym(Mat<eT>& out, const Base<eT,T1>& X, const uword layout)
  {
  arma_extra_debug_sigprint();
  
  out = X.get_ref();
  
  arma_debug_check( (out.is_square() == false), "inv(): given matrix is not square" );
  
  if(out.is_empty())
    {
    return true;
    }
  
  bool status;
  
  #if defined(ARMA_USE_LAPACK)
    {
    arma_debug_assert_blas_size(out);
    
    char     uplo  = (layout == 0) ? 'U' : 'L';
    blas_int n     = blas_int(out.n_rows);
    blas_int lwork = (std::max)(blas_int(podarray_prealloc_n_elem::val), 2*n);
    blas_int info  = 0;
    
    podarray<blas_int> ipiv;
    ipiv.set_size(out.n_rows);
    
    podarray<eT> work;
    work.set_size( uword(lwork) );
    
    lapack::sytrf(&uplo, &n, out.memptr(), &n, ipiv.memptr(), work.memptr(), &lwork, &info);
    
    status = (info == 0);
    
    if(status == true)
      {
      lapack::sytri(&uplo, &n, out.memptr(), &n, ipiv.memptr(), work.memptr(), &info);
      
      out = (layout == 0) ? symmatu(out) : symmatl(out);
      
      status = (info == 0);
      }
    }
  #else
    {
    arma_ignore(layout);
    arma_stop("inv(): use of LAPACK needs to be enabled");
    status = false;
    }
  #endif
  
  return status;
  }



template<typename eT, typename T1>
inline
bool
auxlib::inv_sympd(Mat<eT>& out, const Base<eT,T1>& X, const uword layout)
  {
  arma_extra_debug_sigprint();
  
  out = X.get_ref();
  
  arma_debug_check( (out.is_square() == false), "inv_sympd(): given matrix is not square" );
  
  if(out.is_empty())
    {
    return true;
    }
  
  bool status;
  
  #if defined(ARMA_USE_LAPACK)
    {
    arma_debug_assert_blas_size(out);
    
    char     uplo = (layout == 0) ? 'U' : 'L';
    blas_int n    = blas_int(out.n_rows);
    blas_int info = 0;
    
    lapack::potrf(&uplo, &n, out.memptr(), &n, &info);
    
    status = (info == 0);
    
    if(status == true)
      {
      lapack::potri(&uplo, &n, out.memptr(), &n, &info);
    
      out = (layout == 0) ? symmatu(out) : symmatl(out);
    
      status = (info == 0);
      }
    }
  #else
    {
    arma_ignore(layout);
    arma_stop("inv_sympd(): use of LAPACK needs to be enabled");
    status = false;
    }
  #endif
  
  return status;
  }



template<typename eT, typename T1>
inline
eT
auxlib::det(const Base<eT,T1>& X, const bool slow)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  const bool make_copy = (is_Mat<T1>::value == true) ? true : false;
  
  const unwrap<T1>   tmp(X.get_ref());
  const Mat<eT>& A = tmp.M;
  
  arma_debug_check( (A.is_square() == false), "det(): matrix is not square" );
  
  const uword N = A.n_rows;
  
  if( (N <= 4) && (slow == false) )
    {
    const  T det_min = (is_float<T>::value) ? T(1e-19) : T(1e-154);
    const eT det_val = auxlib::det_tinymat(A, N);
    
    return (std::abs(det_val) >= det_min) ? det_val : auxlib::det_lapack(A, make_copy);
    }
  else
    {
    return auxlib::det_lapack(A, make_copy);
    }
  }



template<typename eT>
inline
eT
auxlib::det_tinymat(const Mat<eT>& X, const uword N)
  {
  arma_extra_debug_sigprint();
  
  switch(N)
    {
    case 0:
      return eT(1);
      break;
    
    case 1:
      return X[0];
      break;
    
    case 2:
      {
      const eT* Xm = X.memptr();
      
      return ( Xm[pos<0,0>::n2]*Xm[pos<1,1>::n2] - Xm[pos<0,1>::n2]*Xm[pos<1,0>::n2] );
      }
      break;
    
    case 3:
      {
      // const double tmp1 = X.at(0,0) * X.at(1,1) * X.at(2,2);
      // const double tmp2 = X.at(0,1) * X.at(1,2) * X.at(2,0);
      // const double tmp3 = X.at(0,2) * X.at(1,0) * X.at(2,1);
      // const double tmp4 = X.at(2,0) * X.at(1,1) * X.at(0,2);
      // const double tmp5 = X.at(2,1) * X.at(1,2) * X.at(0,0);
      // const double tmp6 = X.at(2,2) * X.at(1,0) * X.at(0,1);
      // return (tmp1+tmp2+tmp3) - (tmp4+tmp5+tmp6);
      
      const eT* Xm = X.memptr();
      
      const eT val1 = Xm[pos<0,0>::n3]*(Xm[pos<2,2>::n3]*Xm[pos<1,1>::n3] - Xm[pos<2,1>::n3]*Xm[pos<1,2>::n3]);
      const eT val2 = Xm[pos<1,0>::n3]*(Xm[pos<2,2>::n3]*Xm[pos<0,1>::n3] - Xm[pos<2,1>::n3]*Xm[pos<0,2>::n3]);
      const eT val3 = Xm[pos<2,0>::n3]*(Xm[pos<1,2>::n3]*Xm[pos<0,1>::n3] - Xm[pos<1,1>::n3]*Xm[pos<0,2>::n3]);
      
      return ( val1 - val2 + val3 );
      }
      break;
    
    case 4:
      {
      const eT* Xm = X.memptr();
      
      const eT val = \
          Xm[pos<0,3>::n4] * Xm[pos<1,2>::n4] * Xm[pos<2,1>::n4] * Xm[pos<3,0>::n4] \
        - Xm[pos<0,2>::n4] * Xm[pos<1,3>::n4] * Xm[pos<2,1>::n4] * Xm[pos<3,0>::n4] \
        - Xm[pos<0,3>::n4] * Xm[pos<1,1>::n4] * Xm[pos<2,2>::n4] * Xm[pos<3,0>::n4] \
        + Xm[pos<0,1>::n4] * Xm[pos<1,3>::n4] * Xm[pos<2,2>::n4] * Xm[pos<3,0>::n4] \
        + Xm[pos<0,2>::n4] * Xm[pos<1,1>::n4] * Xm[pos<2,3>::n4] * Xm[pos<3,0>::n4] \
        - Xm[pos<0,1>::n4] * Xm[pos<1,2>::n4] * Xm[pos<2,3>::n4] * Xm[pos<3,0>::n4] \
        - Xm[pos<0,3>::n4] * Xm[pos<1,2>::n4] * Xm[pos<2,0>::n4] * Xm[pos<3,1>::n4] \
        + Xm[pos<0,2>::n4] * Xm[pos<1,3>::n4] * Xm[pos<2,0>::n4] * Xm[pos<3,1>::n4] \
        + Xm[pos<0,3>::n4] * Xm[pos<1,0>::n4] * Xm[pos<2,2>::n4] * Xm[pos<3,1>::n4] \
        - Xm[pos<0,0>::n4] * Xm[pos<1,3>::n4] * Xm[pos<2,2>::n4] * Xm[pos<3,1>::n4] \
        - Xm[pos<0,2>::n4] * Xm[pos<1,0>::n4] * Xm[pos<2,3>::n4] * Xm[pos<3,1>::n4] \
        + Xm[pos<0,0>::n4] * Xm[pos<1,2>::n4] * Xm[pos<2,3>::n4] * Xm[pos<3,1>::n4] \
        + Xm[pos<0,3>::n4] * Xm[pos<1,1>::n4] * Xm[pos<2,0>::n4] * Xm[pos<3,2>::n4] \
        - Xm[pos<0,1>::n4] * Xm[pos<1,3>::n4] * Xm[pos<2,0>::n4] * Xm[pos<3,2>::n4] \
        - Xm[pos<0,3>::n4] * Xm[pos<1,0>::n4] * Xm[pos<2,1>::n4] * Xm[pos<3,2>::n4] \
        + Xm[pos<0,0>::n4] * Xm[pos<1,3>::n4] * Xm[pos<2,1>::n4] * Xm[pos<3,2>::n4] \
        + Xm[pos<0,1>::n4] * Xm[pos<1,0>::n4] * Xm[pos<2,3>::n4] * Xm[pos<3,2>::n4] \
        - Xm[pos<0,0>::n4] * Xm[pos<1,1>::n4] * Xm[pos<2,3>::n4] * Xm[pos<3,2>::n4] \
        - Xm[pos<0,2>::n4] * Xm[pos<1,1>::n4] * Xm[pos<2,0>::n4] * Xm[pos<3,3>::n4] \
        + Xm[pos<0,1>::n4] * Xm[pos<1,2>::n4] * Xm[pos<2,0>::n4] * Xm[pos<3,3>::n4] \
        + Xm[pos<0,2>::n4] * Xm[pos<1,0>::n4] * Xm[pos<2,1>::n4] * Xm[pos<3,3>::n4] \
        - Xm[pos<0,0>::n4] * Xm[pos<1,2>::n4] * Xm[pos<2,1>::n4] * Xm[pos<3,3>::n4] \
        - Xm[pos<0,1>::n4] * Xm[pos<1,0>::n4] * Xm[pos<2,2>::n4] * Xm[pos<3,3>::n4] \
        + Xm[pos<0,0>::n4] * Xm[pos<1,1>::n4] * Xm[pos<2,2>::n4] * Xm[pos<3,3>::n4] \
        ;
      
      return val;
      }
      break;
    
    default:
      return eT(0);
      ;
    }
  }



//! immediate determinant of a matrix using ATLAS or LAPACK
template<typename eT>
inline
eT
auxlib::det_lapack(const Mat<eT>& X, const bool make_copy)
  {
  arma_extra_debug_sigprint();
  
  Mat<eT> X_copy;
  
  if(make_copy == true)
    {
    X_copy = X;
    }
  
  Mat<eT>& tmp = (make_copy == true) ? X_copy : const_cast< Mat<eT>& >(X);
  
  if(tmp.is_empty())
    {
    return eT(1);
    }
  
  
  #if defined(ARMA_USE_ATLAS)
    {
    arma_debug_assert_atlas_size(tmp);
    
    podarray<int> ipiv(tmp.n_rows);
    
    //const int info =
    atlas::clapack_getrf(atlas::CblasColMajor, tmp.n_rows, tmp.n_cols, tmp.memptr(), tmp.n_rows, ipiv.memptr());
    
    // on output tmp appears to be L+U_alt, where U_alt is U with the main diagonal set to zero
    eT val = tmp.at(0,0);
    for(uword i=1; i < tmp.n_rows; ++i)
      {
      val *= tmp.at(i,i);
      }
    
    int sign = +1;
    for(uword i=0; i < tmp.n_rows; ++i)
      {
      if( int(i) != ipiv.mem[i] )  // NOTE: no adjustment required, as the clapack version of getrf() assumes counting from 0
        {
        sign *= -1;
        }
      }
    
    return ( (sign < 0) ? -val : val );
    }
  #elif defined(ARMA_USE_LAPACK)
    {
    arma_debug_assert_blas_size(tmp);
    
    podarray<blas_int> ipiv(tmp.n_rows);
    
    blas_int info   = 0;
    blas_int n_rows = blas_int(tmp.n_rows);
    blas_int n_cols = blas_int(tmp.n_cols);
    
    lapack::getrf(&n_rows, &n_cols, tmp.memptr(), &n_rows, ipiv.memptr(), &info);
    
    // on output tmp appears to be L+U_alt, where U_alt is U with the main diagonal set to zero
    eT val = tmp.at(0,0);
    for(uword i=1; i < tmp.n_rows; ++i)
      {
      val *= tmp.at(i,i);
      }
    
    blas_int sign = +1;
    for(uword i=0; i < tmp.n_rows; ++i)
      {
      if( blas_int(i) != (ipiv.mem[i] - 1) )  // NOTE: adjustment of -1 is required as Fortran counts from 1
        {
        sign *= -1;
        }
      }
    
    return ( (sign < 0) ? -val : val );
    }
  #else
    {
    arma_ignore(X);
    arma_ignore(make_copy);
    arma_ignore(tmp);
    arma_stop("det(): use of ATLAS or LAPACK needs to be enabled");
    return eT(0);
    }
  #endif
  }



//! immediate log determinant of a matrix using ATLAS or LAPACK
template<typename eT, typename T1>
inline
bool
auxlib::log_det(eT& out_val, typename get_pod_type<eT>::result& out_sign, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  #if defined(ARMA_USE_ATLAS)
    {
    Mat<eT> tmp(X.get_ref());
    arma_debug_check( (tmp.is_square() == false), "log_det(): given matrix is not square" );
    
    if(tmp.is_empty())
      {
      out_val  = eT(0);
      out_sign =  T(1);
      return true;
      }
    
    arma_debug_assert_atlas_size(tmp);
    
    podarray<int> ipiv(tmp.n_rows);
    
    const int info = atlas::clapack_getrf(atlas::CblasColMajor, tmp.n_rows, tmp.n_cols, tmp.memptr(), tmp.n_rows, ipiv.memptr());
    
    // on output tmp appears to be L+U_alt, where U_alt is U with the main diagonal set to zero
    
    sword sign = (is_complex<eT>::value == false) ? ( (access::tmp_real( tmp.at(0,0) ) < T(0)) ? -1 : +1 ) : +1;
    eT   val = (is_complex<eT>::value == false) ? std::log( (access::tmp_real( tmp.at(0,0) ) < T(0)) ? tmp.at(0,0)*T(-1) : tmp.at(0,0) ) : std::log( tmp.at(0,0) );
    
    for(uword i=1; i < tmp.n_rows; ++i)
      {
      const eT x = tmp.at(i,i);
      
      sign *= (is_complex<eT>::value == false) ? ( (access::tmp_real(x) < T(0)) ? -1 : +1 ) : +1;
      val  += (is_complex<eT>::value == false) ? std::log( (access::tmp_real(x) < T(0)) ? x*T(-1) : x ) : std::log(x);
      }
    
    for(uword i=0; i < tmp.n_rows; ++i)
      {
      if( int(i) != ipiv.mem[i] )  // NOTE: no adjustment required, as the clapack version of getrf() assumes counting from 0
        {
        sign *= -1;
        }
      }
    
    out_val  = val;
    out_sign = T(sign);
    
    return (info == 0);
    }
  #elif defined(ARMA_USE_LAPACK)
    {
    Mat<eT> tmp(X.get_ref());
    arma_debug_check( (tmp.is_square() == false), "log_det(): given matrix is not square" );
    
    if(tmp.is_empty())
      {
      out_val  = eT(0);
      out_sign =  T(1);
      return true;
      }
    
    arma_debug_assert_blas_size(tmp);
    
    podarray<blas_int> ipiv(tmp.n_rows);
    
    blas_int info   = 0;
    blas_int n_rows = blas_int(tmp.n_rows);
    blas_int n_cols = blas_int(tmp.n_cols);
    
    lapack::getrf(&n_rows, &n_cols, tmp.memptr(), &n_rows, ipiv.memptr(), &info);
    
    // on output tmp appears to be L+U_alt, where U_alt is U with the main diagonal set to zero
    
    sword sign = (is_complex<eT>::value == false) ? ( (access::tmp_real( tmp.at(0,0) ) < T(0)) ? -1 : +1 ) : +1;
    eT   val = (is_complex<eT>::value == false) ? std::log( (access::tmp_real( tmp.at(0,0) ) < T(0)) ? tmp.at(0,0)*T(-1) : tmp.at(0,0) ) : std::log( tmp.at(0,0) );
    
    for(uword i=1; i < tmp.n_rows; ++i)
      {
      const eT x = tmp.at(i,i);
      
      sign *= (is_complex<eT>::value == false) ? ( (access::tmp_real(x) < T(0)) ? -1 : +1 ) : +1;
      val  += (is_complex<eT>::value == false) ? std::log( (access::tmp_real(x) < T(0)) ? x*T(-1) : x ) : std::log(x);
      }
    
    for(uword i=0; i < tmp.n_rows; ++i)
      {
      if( blas_int(i) != (ipiv.mem[i] - 1) )  // NOTE: adjustment of -1 is required as Fortran counts from 1
        {
        sign *= -1;
        }
      }
    
    out_val  = val;
    out_sign = T(sign);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(X);
    
    out_val  = eT(0);
    out_sign =  T(0);
    
    arma_stop("log_det(): use of ATLAS or LAPACK needs to be enabled");
    
    return false;
    }
  #endif
  }



//! immediate LU decomposition of a matrix using ATLAS or LAPACK
template<typename eT, typename T1>
inline
bool
auxlib::lu(Mat<eT>& L, Mat<eT>& U, podarray<blas_int>& ipiv, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  U = X.get_ref();
  
  const uword U_n_rows = U.n_rows;
  const uword U_n_cols = U.n_cols;
  
  if(U.is_empty())
    {
    L.set_size(U_n_rows, 0);
    U.set_size(0, U_n_cols);
    ipiv.reset();
    return true;
    }
  
  #if defined(ARMA_USE_ATLAS) || defined(ARMA_USE_LAPACK)
    {
    bool status;
    
    #if defined(ARMA_USE_ATLAS)
      {
      arma_debug_assert_atlas_size(U);
      
      ipiv.set_size( (std::min)(U_n_rows, U_n_cols) );
      
      int info = atlas::clapack_getrf(atlas::CblasColMajor, U_n_rows, U_n_cols, U.memptr(), U_n_rows, ipiv.memptr());
      
      status = (info == 0);
      }
    #elif defined(ARMA_USE_LAPACK)
      {
      arma_debug_assert_blas_size(U);
      
      ipiv.set_size( (std::min)(U_n_rows, U_n_cols) );
      
      blas_int info = 0;
      
      blas_int n_rows = U_n_rows;
      blas_int n_cols = U_n_cols;
      
      
      lapack::getrf(&n_rows, &n_cols, U.memptr(), &n_rows, ipiv.memptr(), &info);
      
      // take into account that Fortran counts from 1
      arrayops::inplace_minus(ipiv.memptr(), blas_int(1), ipiv.n_elem);
      
      status = (info == 0);
      }
    #endif
    
    L.copy_size(U);
    
    for(uword col=0; col < U_n_cols; ++col)
      {
      for(uword row=0; (row < col) && (row < U_n_rows); ++row)
        {
        L.at(row,col) = eT(0);
        }
      
      if( L.in_range(col,col) == true )
        {
        L.at(col,col) = eT(1);
        }
      
      for(uword row = (col+1); row < U_n_rows; ++row)
        {
        L.at(row,col) = U.at(row,col);
        U.at(row,col) = eT(0);
        }
      }
    
    return status;
    }
  #else
    {
    arma_stop("lu(): use of ATLAS or LAPACK needs to be enabled");
    
    return false;
    }
  #endif
  }



template<typename eT, typename T1>
inline
bool
auxlib::lu(Mat<eT>& L, Mat<eT>& U, Mat<eT>& P, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  podarray<blas_int> ipiv1;
  const bool status = auxlib::lu(L, U, ipiv1, X);
  
  if(status == true)
    {
    if(U.is_empty())
      {
      // L and U have been already set to the correct empty matrices
      P.eye(L.n_rows, L.n_rows);
      return true;
      }
    
    const uword n      = ipiv1.n_elem;
    const uword P_rows = U.n_rows;
    
    podarray<blas_int> ipiv2(P_rows);
    
    const blas_int* ipiv1_mem = ipiv1.memptr();
          blas_int* ipiv2_mem = ipiv2.memptr();
    
    for(uword i=0; i<P_rows; ++i)
      {
      ipiv2_mem[i] = blas_int(i);
      }
    
    for(uword i=0; i<n; ++i)
      {
      const uword k = static_cast<uword>(ipiv1_mem[i]);
      
      if( ipiv2_mem[i] != ipiv2_mem[k] )
        {
        std::swap( ipiv2_mem[i], ipiv2_mem[k] );
        }
      }
    
    P.zeros(P_rows, P_rows);
    
    for(uword row=0; row<P_rows; ++row)
      {
      P.at(row, static_cast<uword>(ipiv2_mem[row])) = eT(1);
      }
    
    if(L.n_cols > U.n_rows)
      {
      L.shed_cols(U.n_rows, L.n_cols-1);
      }
      
    if(U.n_rows > L.n_cols)
      {
      U.shed_rows(L.n_cols, U.n_rows-1);
      }
    }
  
  return status;
  }



template<typename eT, typename T1>
inline
bool
auxlib::lu(Mat<eT>& L, Mat<eT>& U, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  podarray<blas_int> ipiv1;
  const bool status = auxlib::lu(L, U, ipiv1, X);
  
  if(status == true)
    {
    if(U.is_empty())
      {
      // L and U have been already set to the correct empty matrices
      return true;
      }
    
    const uword n      = ipiv1.n_elem;
    const uword P_rows = U.n_rows;
    
    podarray<blas_int> ipiv2(P_rows);
    
    const blas_int* ipiv1_mem = ipiv1.memptr();
          blas_int* ipiv2_mem = ipiv2.memptr();
    
    for(uword i=0; i<P_rows; ++i)
      {
      ipiv2_mem[i] = blas_int(i);
      }
    
    for(uword i=0; i<n; ++i)
      {
      const uword k = static_cast<uword>(ipiv1_mem[i]);
      
      if( ipiv2_mem[i] != ipiv2_mem[k] )
        {
        std::swap( ipiv2_mem[i], ipiv2_mem[k] );
        L.swap_rows( static_cast<uword>(ipiv2_mem[i]), static_cast<uword>(ipiv2_mem[k]) );
        }
      }
    
    if(L.n_cols > U.n_rows)
      {
      L.shed_cols(U.n_rows, L.n_cols-1);
      }
      
    if(U.n_rows > L.n_cols)
      {
      U.shed_rows(L.n_cols, U.n_rows-1);
      }
    }
  
  return status;
  }



//! immediate eigenvalues of a symmetric real matrix using LAPACK
template<typename eT, typename T1>
inline
bool
auxlib::eig_sym(Col<eT>& eigval, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    Mat<eT> A(X.get_ref());
    
    arma_debug_check( (A.is_square() == false), "eig_sym(): given matrix is not square");
    
    if(A.is_empty())
      {
      eigval.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    eigval.set_size(A.n_rows);
    
    char jobz  = 'N';
    char uplo  = 'U';
    
    blas_int N     = blas_int(A.n_rows);
    blas_int lwork = 3 * ( (std::max)(blas_int(1), 3*N-1) );
    blas_int info  = 0;
    
    podarray<eT> work( static_cast<uword>(lwork) );
    
    lapack::syev(&jobz, &uplo, &N, A.memptr(), &N, eigval.memptr(), work.memptr(), &lwork, &info);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(eigval);
    arma_ignore(X);
    arma_stop("eig_sym(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//! immediate eigenvalues of a hermitian complex matrix using LAPACK
template<typename T, typename T1>
inline
bool
auxlib::eig_sym(Col<T>& eigval, const Base<std::complex<T>,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    typedef typename std::complex<T> eT;
    
    Mat<eT> A(X.get_ref());
    
    arma_debug_check( (A.is_square() == false), "eig_sym(): given matrix is not square");
    
    if(A.is_empty())
      {
      eigval.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    eigval.set_size(A.n_rows);
    
    char jobz  = 'N'; 
    char uplo  = 'U';
    
    blas_int N     = blas_int(A.n_rows);
    blas_int lwork = 3 * ( (std::max)(blas_int(1), 2*N-1) );
    blas_int info  = 0;
    
    podarray<eT>  work( static_cast<uword>(lwork) );
    podarray<T>  rwork( static_cast<uword>( (std::max)(blas_int(1), 3*N-2) ) );
    
    arma_extra_debug_print("lapack::heev()");
    lapack::heev(&jobz, &uplo, &N, A.memptr(), &N, eigval.memptr(), work.memptr(), &lwork, rwork.memptr(), &info);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(eigval);
    arma_ignore(X);
    arma_stop("eig_sym(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//! immediate eigenvalues and eigenvectors of a symmetric real matrix using LAPACK
template<typename eT, typename T1>
inline
bool
auxlib::eig_sym(Col<eT>& eigval, Mat<eT>& eigvec, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    eigvec = X.get_ref();
    
    arma_debug_check( (eigvec.is_square() == false), "eig_sym(): given matrix is not square" );
    
    if(eigvec.is_empty())
      {
      eigval.reset();
      eigvec.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(eigvec);
    
    eigval.set_size(eigvec.n_rows);
    
    char jobz  = 'V';
    char uplo  = 'U';
    
    blas_int N     = blas_int(eigvec.n_rows);
    blas_int lwork = 3 * ( (std::max)(blas_int(1), 3*N-1) );
    blas_int info  = 0;
    
    podarray<eT> work( static_cast<uword>(lwork) );
    
    lapack::syev(&jobz, &uplo, &N, eigvec.memptr(), &N, eigval.memptr(), work.memptr(), &lwork, &info);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(eigval);
    arma_ignore(eigvec);
    arma_ignore(X);
    arma_stop("eig_sym(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//! immediate eigenvalues and eigenvectors of a hermitian complex matrix using LAPACK
template<typename T, typename T1>
inline
bool
auxlib::eig_sym(Col<T>& eigval, Mat< std::complex<T> >& eigvec, const Base<std::complex<T>,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    typedef typename std::complex<T> eT;
    
    eigvec = X.get_ref();
    
    arma_debug_check( (eigvec.is_square() == false), "eig_sym(): given matrix is not square" );
    
    if(eigvec.is_empty())
      {
      eigval.reset();
      eigvec.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(eigvec);
    
    eigval.set_size(eigvec.n_rows);
    
    char jobz  = 'V';
    char uplo  = 'U';
    
    blas_int N     = blas_int(eigvec.n_rows);
    blas_int lwork = 3 * ( (std::max)(blas_int(1), 2*N-1) );
    blas_int info  = 0;
    
    podarray<eT>  work( static_cast<uword>(lwork) );
    podarray<T>  rwork( static_cast<uword>((std::max)(blas_int(1), 3*N-2)) );
    
    arma_extra_debug_print("lapack::heev()");
    lapack::heev(&jobz, &uplo, &N, eigvec.memptr(), &N, eigval.memptr(), work.memptr(), &lwork, rwork.memptr(), &info);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(eigval);
    arma_ignore(eigvec);
    arma_ignore(X);
    arma_stop("eig_sym(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//! immediate eigenvalues and eigenvectors of a symmetric real matrix using LAPACK (divide and conquer algorithm)
template<typename eT, typename T1>
inline
bool
auxlib::eig_sym_dc(Col<eT>& eigval, Mat<eT>& eigvec, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    eigvec = X.get_ref();
    
    arma_debug_check( (eigvec.is_square() == false), "eig_sym(): given matrix is not square" );
    
    if(eigvec.is_empty())
      {
      eigval.reset();
      eigvec.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(eigvec);
    
    eigval.set_size(eigvec.n_rows);
    
    char jobz = 'V';
    char uplo = 'U';
    
    blas_int N      = blas_int(eigvec.n_rows);
    blas_int lwork  = 2 * (1 + 6*N + 2*(N*N));
    blas_int liwork = 3 * (3 + 5*N);
    blas_int info   = 0;
    
    podarray<eT>        work( static_cast<uword>( lwork) );
    podarray<blas_int> iwork( static_cast<uword>(liwork) ); 
    
    arma_extra_debug_print("lapack::syevd()");
    lapack::syevd(&jobz, &uplo, &N, eigvec.memptr(), &N, eigval.memptr(), work.memptr(), &lwork, iwork.memptr(), &liwork, &info);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(eigval);
    arma_ignore(eigvec);
    arma_ignore(X);
    arma_stop("eig_sym(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//! immediate eigenvalues and eigenvectors of a hermitian complex matrix using LAPACK (divide and conquer algorithm)
template<typename T, typename T1>
inline
bool
auxlib::eig_sym_dc(Col<T>& eigval, Mat< std::complex<T> >& eigvec, const Base<std::complex<T>,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    typedef typename std::complex<T> eT;
    
    eigvec = X.get_ref();
    
    arma_debug_check( (eigvec.is_square() == false), "eig_sym(): given matrix is not square" );
    
    if(eigvec.is_empty())
      {
      eigval.reset();
      eigvec.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(eigvec);
    
    eigval.set_size(eigvec.n_rows);
    
    char jobz  = 'V';
    char uplo  = 'U';
    
    blas_int N      = blas_int(eigvec.n_rows);
    blas_int lwork  = 2 * (2*N + N*N);
    blas_int lrwork = 2 * (1 + 5*N + 2*(N*N));
    blas_int liwork = 3 * (3 + 5*N);
    blas_int info   = 0;
    
    podarray<eT>        work( static_cast<uword>(lwork)  );
    podarray<T>        rwork( static_cast<uword>(lrwork) );
    podarray<blas_int> iwork( static_cast<uword>(liwork) ); 
    
    arma_extra_debug_print("lapack::heevd()");
    lapack::heevd(&jobz, &uplo, &N, eigvec.memptr(), &N, eigval.memptr(), work.memptr(), &lwork, rwork.memptr(), &lrwork, iwork.memptr(), &liwork, &info);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(eigval);
    arma_ignore(eigvec);
    arma_ignore(X);
    arma_stop("eig_sym(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//! Eigenvalues and eigenvectors of a general square real matrix using LAPACK.
//! The argument 'side' specifies which eigenvectors should be calculated
//! (see code for mode details).
template<typename T, typename T1>
inline
bool
auxlib::eig_gen
  (
  Col< std::complex<T> >&   eigval,
  Mat<T>&                 l_eigvec,
  Mat<T>&                 r_eigvec,
  const Base<T,T1>&       X,
  const char              side
  )
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    char jobvl;
    char jobvr;
    
    switch(side)
      {
      case 'l':  // left
        jobvl = 'V';
        jobvr = 'N';
        break;
        
      case 'r':  // right
        jobvl = 'N';
        jobvr = 'V';
        break;
        
      case 'b':  // both
        jobvl = 'V';
        jobvr = 'V';
        break;
        
      case 'n':  // neither
        jobvl = 'N';
        jobvr = 'N';
        break;
      
      default:
        arma_stop("eig_gen(): parameter 'side' is invalid");
        return false;
      }
    
    Mat<T> A(X.get_ref());
    arma_debug_check( (A.is_square() == false), "eig_gen(): given matrix is not square" );
    
    if(A.is_empty())
      {
      eigval.reset();
      l_eigvec.reset();
      r_eigvec.reset();
      return true;
      }
    
    if(A.is_finite() == false)  { return false; }  // workaround for a bug in LAPACK 3.5
    
    arma_debug_assert_blas_size(A);
    
    const uword A_n_rows = A.n_rows;
    
    eigval.set_size(A_n_rows);
    
    l_eigvec.set_size( ((jobvl == 'V') ? A_n_rows : 1), A_n_rows );
    r_eigvec.set_size( ((jobvr == 'V') ? A_n_rows : 1), A_n_rows );
    
    blas_int N     = blas_int(A_n_rows);
    blas_int ldvl  = blas_int(l_eigvec.n_rows);
    blas_int ldvr  = blas_int(r_eigvec.n_rows);
    blas_int lwork = 3 * ( (std::max)(blas_int(1), 4*N) );
    blas_int info  = 0;
    
    podarray<T> work( static_cast<uword>(lwork) );
    
    podarray<T> wr(A_n_rows);
    podarray<T> wi(A_n_rows);
    
    arma_extra_debug_print("lapack::geev()");
    lapack::geev(&jobvl, &jobvr, &N, A.memptr(), &N, wr.memptr(), wi.memptr(), l_eigvec.memptr(), &ldvl, r_eigvec.memptr(), &ldvr, work.memptr(), &lwork, &info);
    
    eigval.set_size(A_n_rows);
    for(uword i=0; i<A_n_rows; ++i)
      {
      eigval[i] = std::complex<T>(wr[i], wi[i]);
      }
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(eigval);
    arma_ignore(l_eigvec);
    arma_ignore(r_eigvec);
    arma_ignore(X);
    arma_ignore(side);
    arma_stop("eig_gen(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }





//! Eigenvalues and eigenvectors of a general square complex matrix using LAPACK
//! The argument 'side' specifies which eigenvectors should be calculated
//! (see code for mode details).
template<typename T, typename T1>
inline
bool
auxlib::eig_gen
  (
  Col< std::complex<T> >&              eigval,
  Mat< std::complex<T> >&            l_eigvec, 
  Mat< std::complex<T> >&            r_eigvec, 
  const Base< std::complex<T>, T1 >& X, 
  const char                         side
  )
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    typedef typename std::complex<T> eT;
    
    char jobvl;
    char jobvr;
    
    switch(side)
      {
      case 'l':  // left
        jobvl = 'V';
        jobvr = 'N';
        break;
        
      case 'r':  // right
        jobvl = 'N';
        jobvr = 'V';
        break;
        
      case 'b':  // both
        jobvl = 'V';
        jobvr = 'V';
        break;
        
      case 'n':  // neither
        jobvl = 'N';
        jobvr = 'N';
        break;
      
      default:
        arma_stop("eig_gen(): parameter 'side' is invalid");
        return false;
      }
    
    Mat<eT> A(X.get_ref());
    arma_debug_check( (A.is_square() == false), "eig_gen(): given matrix is not square" );
    
    if(A.is_empty())
      {
      eigval.reset();
      l_eigvec.reset();
      r_eigvec.reset();
      return true;
      }
    
    if(A.is_finite() == false)  { return false; }  // workaround for a bug in LAPACK 3.5
    
    arma_debug_assert_blas_size(A);
    
    const uword A_n_rows = A.n_rows;
    
    eigval.set_size(A_n_rows);
    
    l_eigvec.set_size( ((jobvl == 'V') ? A_n_rows : 1), A_n_rows );
    r_eigvec.set_size( ((jobvr == 'V') ? A_n_rows : 1), A_n_rows );
    
    blas_int N     = blas_int(A_n_rows);
    blas_int ldvl  = blas_int(l_eigvec.n_rows);
    blas_int ldvr  = blas_int(r_eigvec.n_rows);
    blas_int lwork = 3 * ( (std::max)(blas_int(1), 2*N) );
    blas_int info  = 0;
    
    podarray<eT>  work( static_cast<uword>(lwork) );
    podarray<T>  rwork( static_cast<uword>(2*N)   );
    
    arma_extra_debug_print("lapack::cx_geev()");
    lapack::cx_geev(&jobvl, &jobvr, &N, A.memptr(), &N, eigval.memptr(), l_eigvec.memptr(), &ldvl, r_eigvec.memptr(), &ldvr, work.memptr(), &lwork, rwork.memptr(), &info);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(eigval);
    arma_ignore(l_eigvec);
    arma_ignore(r_eigvec);
    arma_ignore(X);
    arma_ignore(side);
    arma_stop("eig_gen(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//! Eigenvalues and eigenvectors of general square real matrix pair.
//! The argument 'side' specifies which eigenvectors should be calculated.
template<typename T, typename T1, typename T2>
inline
bool
auxlib::eig_pair
  (
  Col< std::complex<T> >&   eigval,
  Mat<T>&                 l_eigvec,
  Mat<T>&                 r_eigvec,
  const Base<T,T1>&       X,
  const Base<T,T2>&       Y,
  const char              side
  )
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    char jobvl;
    char jobvr;
    
    switch(side)
      {
      case 'l':  // left
        jobvl = 'V';
        jobvr = 'N';
        break;
        
      case 'r':  // right
        jobvl = 'N';
        jobvr = 'V';
        break;
        
      case 'b':  // both
        jobvl = 'V';
        jobvr = 'V';
        break;
        
      case 'n':  // neither
        jobvl = 'N';
        jobvr = 'N';
        break;
      
      default:
        arma_stop("eig_pair(): parameter 'side' is invalid");
        return false;
      }
    
    Mat<T> A(X.get_ref());
    Mat<T> B(Y.get_ref());
    
    arma_debug_check( ((A.is_square() == false) || (B.is_square() == false)), "eig_pair(): given matrix is not square" );
    
    arma_debug_check( (A.n_rows != B.n_rows), "eig_pair(): given matrices must have the same size" );
    
    if(A.is_empty())
      {
      eigval.reset();
      l_eigvec.reset();
      r_eigvec.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    const uword A_n_rows = A.n_rows;
    
    eigval.set_size(A_n_rows);
    
    l_eigvec.set_size( ((jobvl == 'V') ? A_n_rows : 1), A_n_rows );
    r_eigvec.set_size( ((jobvr == 'V') ? A_n_rows : 1), A_n_rows );
    
    blas_int N     = blas_int(A_n_rows);
    blas_int ldvl  = blas_int(l_eigvec.n_rows);
    blas_int ldvr  = blas_int(r_eigvec.n_rows);
    blas_int lwork = 3 * ( (std::max)(blas_int(1), 8*N) );
    blas_int info  = 0;
    
    podarray<T> work( static_cast<uword>(lwork) );
    
    podarray<T> alphar(A_n_rows);
    podarray<T> alphai(A_n_rows);
    podarray<T>   beta(A_n_rows);
    
    arma_extra_debug_print("lapack::ggev()");
    lapack::ggev
      (
      &jobvl, &jobvr, &N,
      A.memptr(), &N,  B.memptr(), &N,
      alphar.memptr(), alphai.memptr(), beta.memptr(),
      l_eigvec.memptr(), &ldvl, r_eigvec.memptr(), &ldvr,
      work.memptr(), &lwork,
      &info
      );
    
    if(info == 0)
      {
      // from LAPACK docs:
      // If ALPHAI(j) is zero, then the j-th eigenvalue is real;
      // if positive, then the j-th and (j+1)-st eigenvalues are a complex conjugate pair,
      // with ALPHAI(j+1) negative.
      
      eigval.set_size(A_n_rows);
      
      const T* alphar_mem = alphar.memptr();
      const T* alphai_mem = alphai.memptr();
      const T*   beta_mem =   beta.memptr();
      
      bool beta_has_zero = false;
      
      for(uword j=0; j<A_n_rows; ++j)
        {
        const T alphai_val = alphai_mem[j];
        const T   beta_val =   beta_mem[j];
        
        if(beta_val == T(0))  { beta_has_zero = true; }
        
        const T re = alphar_mem[j] / beta_val;
        const T im = alphai_val    / beta_val;
        
        eigval[j] = std::complex<T>(re, im);
        
        if( (alphai_val > T(0)) && (j < (A_n_rows-1)) )
          {
          // ensure we have exact conjugate
          ++j;
          eigval[j] = std::complex<T>(re,-im);
          }
        }
      
      arma_debug_warn(beta_has_zero, "eig_pair(): warning: given matrix appears ill-conditioned");
      
      return true;
      }
    else
      {
      return false;
      }
    }
  #else
    {
    arma_ignore(eigval);
    arma_ignore(l_eigvec);
    arma_ignore(r_eigvec);
    arma_ignore(X);
    arma_ignore(Y);
    arma_ignore(side);
    arma_stop("eig_pair(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }





//! Eigenvalues and eigenvectors of general square complex matrix pair.
//! The argument 'side' specifies which eigenvectors should be calculated
template<typename T, typename T1, typename T2>
inline
bool
auxlib::eig_pair
  (
  Col< std::complex<T> >&              eigval,
  Mat< std::complex<T> >&            l_eigvec, 
  Mat< std::complex<T> >&            r_eigvec, 
  const Base< std::complex<T>, T1 >& X, 
  const Base< std::complex<T>, T2 >& Y, 
  const char                         side
  )
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    typedef typename std::complex<T> eT;
    
    char jobvl;
    char jobvr;
    
    switch(side)
      {
      case 'l':  // left
        jobvl = 'V';
        jobvr = 'N';
        break;
        
      case 'r':  // right
        jobvl = 'N';
        jobvr = 'V';
        break;
        
      case 'b':  // both
        jobvl = 'V';
        jobvr = 'V';
        break;
        
      case 'n':  // neither
        jobvl = 'N';
        jobvr = 'N';
        break;
      
      default:
        arma_stop("eig_pair(): parameter 'side' is invalid");
        return false;
      }
    
    Mat<eT> A(X.get_ref());
    Mat<eT> B(Y.get_ref());
    
    arma_debug_check( ((A.is_square() == false) || (B.is_square() == false)), "eig_pair(): given matrix is not square" );
    
    arma_debug_check( (A.n_rows != B.n_rows), "eig_pair(): given matrices must have the same size" );
    
    if(A.is_empty())
      {
      eigval.reset();
      l_eigvec.reset();
      r_eigvec.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    const uword A_n_rows = A.n_rows;
    
    podarray<eT> alpha(A_n_rows);
    podarray<eT>  beta(A_n_rows);
    
    l_eigvec.set_size( ((jobvl == 'V') ? A_n_rows : 1), A_n_rows );
    r_eigvec.set_size( ((jobvr == 'V') ? A_n_rows : 1), A_n_rows );
    
    blas_int N     = blas_int(A_n_rows);
    blas_int ldvl  = blas_int(l_eigvec.n_rows);
    blas_int ldvr  = blas_int(r_eigvec.n_rows);
    blas_int lwork = 3 * ((std::max)(blas_int(1),2*N));
    blas_int info  = 0;
    
    podarray<eT>  work( static_cast<uword>(lwork) );
    podarray<T>  rwork( static_cast<uword>(8*N)   );
    
    arma_extra_debug_print("lapack::cx_ggev()");
    lapack::cx_ggev
      (
      &jobvl, &jobvr, &N,
      A.memptr(), &N, B.memptr(), &N,
      alpha.memptr(), beta.memptr(),
      l_eigvec.memptr(), &ldvl, r_eigvec.memptr(), &ldvr,
      work.memptr(), &lwork, rwork.memptr(),
      &info
      );
    
    if(info == 0)
      {
      // TODO: figure out a more robust way to create the eigen values;
      // TODO: from LAPACK docs: the quotients ALPHA(j)/BETA(j) may easily over- or underflow, and BETA(j) may even be zero
      
      eigval.set_size(A_n_rows);
      
            eT* eigval_mem = eigval.memptr();
      const eT*  alpha_mem = alpha.memptr();
      const eT*   beta_mem =  beta.memptr();
      
      const std::complex<T> cx_zero( T(0), T(0) );
      
      bool beta_has_zero = false;
      
      for(uword i=0; i<A_n_rows; ++i)
        {
        eigval_mem[i] = alpha_mem[i] / beta_mem[i];
        
        if(beta_mem[i] == cx_zero)  { beta_has_zero = true; }
        
        // // TODO: not sure if this is the right approach
        // eigval_mem[i] = (beta_mem[i] != cx_zero) ? (alpha_mem[i] / beta_mem[i]) : cx_zero;
        }
      
      arma_debug_warn(beta_has_zero, "eig_pair(): warning: given matrix appears ill-conditioned");
      
      return true;
      }
    else
      {
      return false;
      }
    }
  #else
    {
    arma_ignore(eigval);
    arma_ignore(l_eigvec);
    arma_ignore(r_eigvec);
    arma_ignore(X);
    arma_ignore(Y);
    arma_ignore(side);
    arma_stop("eig_pair(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename eT, typename T1>
inline
bool
auxlib::chol(Mat<eT>& out, const Base<eT,T1>& X, const uword layout)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    out = X.get_ref();
    
    arma_debug_check( (out.is_square() == false), "chol(): given matrix is not square" );
    
    if(out.is_empty())  { return true; }
    
    arma_debug_assert_blas_size(out);
    
    const uword out_n_rows = out.n_rows;
    
    char      uplo = (layout == 0) ? 'U' : 'L';
    blas_int  n    = out_n_rows;
    blas_int  info = 0;
    
    lapack::potrf(&uplo, &n, out.memptr(), &n, &info);
    
    if(layout == 0)
      {
      for(uword col=0; col < out_n_rows; ++col)
        {
        eT* colptr = out.colptr(col);
        
        for(uword row=(col+1); row < out_n_rows; ++row)  { colptr[row] = eT(0); }
        }
      }
    else
      {
      for(uword col=1; col < out_n_rows; ++col)
        {
        eT* colptr = out.colptr(col);
        
        for(uword row=0; row < col; ++row)  { colptr[row] = eT(0); }
        }
      }
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(out);
    arma_ignore(X);
    arma_ignore(layout);
    
    arma_stop("chol(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename eT, typename T1>
inline
bool
auxlib::qr(Mat<eT>& Q, Mat<eT>& R, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    R = X.get_ref();
    
    const uword R_n_rows = R.n_rows;
    const uword R_n_cols = R.n_cols;
    
    if(R.is_empty())
      {
      Q.eye(R_n_rows, R_n_rows);
      return true;
      }
    
    arma_debug_assert_blas_size(R);
    
    blas_int m         = static_cast<blas_int>(R_n_rows);
    blas_int n         = static_cast<blas_int>(R_n_cols);
    blas_int lwork     = 0;
    blas_int lwork_min = (std::max)(blas_int(1), (std::max)(m,n));  // take into account requirements of geqrf() _and_ orgqr()/ungqr()
    blas_int k         = (std::min)(m,n);
    blas_int info      = 0;
    
    podarray<eT> tau( static_cast<uword>(k) );
    
    eT        work_query[2];
    blas_int lwork_query = -1;
    
    lapack::geqrf(&m, &n, R.memptr(), &m, tau.memptr(), &work_query[0], &lwork_query, &info);
    
    if(info == 0)
      {
      const blas_int lwork_proposed = static_cast<blas_int>( access::tmp_real(work_query[0]) );
      
      lwork = (lwork_proposed > lwork_min) ? lwork_proposed : lwork_min;
      }
    else
      {
      return false;
      }
    
    podarray<eT> work( static_cast<uword>(lwork) );
    
    lapack::geqrf(&m, &n, R.memptr(), &m, tau.memptr(), work.memptr(), &lwork, &info);
    
    Q.set_size(R_n_rows, R_n_rows);
    
    arrayops::copy( Q.memptr(), R.memptr(), (std::min)(Q.n_elem, R.n_elem) );
    
    //
    // construct R
    
    for(uword col=0; col < R_n_cols; ++col)
      {
      for(uword row=(col+1); row < R_n_rows; ++row)
        {
        R.at(row,col) = eT(0);
        }
      }
    
    
    if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )
      {
      lapack::orgqr(&m, &m, &k, Q.memptr(), &m, tau.memptr(), work.memptr(), &lwork, &info);
      }
    else
    if( (is_supported_complex_float<eT>::value == true) || (is_supported_complex_double<eT>::value == true) )
      {
      lapack::ungqr(&m, &m, &k, Q.memptr(), &m, tau.memptr(), work.memptr(), &lwork, &info);
      }
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(Q);
    arma_ignore(R);
    arma_ignore(X);
    arma_stop("qr(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename eT, typename T1>
inline
bool 
auxlib::qr_econ(Mat<eT>& Q, Mat<eT>& R, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  // This function implements a memory-efficient QR for a non-square X that has dimensions m x n.
  // This basically discards the basis for the null-space.
  // 
  // if m <= n: (use standard routine)
  //     Q[m,m]*R[m,n] = X[m,n]
  //     geqrf Needs A[m,n]: Uses R
  //     orgqr Needs A[m,m]: Uses Q
  // otherwise: (memory-efficient routine)
  //     Q[m,n]*R[n,n] = X[m,n]
  //     geqrf Needs A[m,n]: Uses Q
  //     geqrf Needs A[m,n]: Uses Q
  
  #if defined(ARMA_USE_LAPACK)
    {
    if(is_Mat<T1>::value == true)
      {
      const unwrap<T1>   tmp(X.get_ref());
      const Mat<eT>& M = tmp.M;
      
      if(M.n_rows < M.n_cols)
        {
        return auxlib::qr(Q, R, X);
        }
      }
    
    Q = X.get_ref();
    
    const uword Q_n_rows = Q.n_rows;
    const uword Q_n_cols = Q.n_cols;
    
    if( Q_n_rows <= Q_n_cols )
      {
      return auxlib::qr(Q, R, Q);
      }
    
    if(Q.is_empty())
      {
      Q.set_size(Q_n_rows, 0       );
      R.set_size(0,        Q_n_cols);
      return true;
      }
    
    arma_debug_assert_blas_size(Q);
    
    blas_int m         = static_cast<blas_int>(Q_n_rows);
    blas_int n         = static_cast<blas_int>(Q_n_cols);
    blas_int lwork     = 0;
    blas_int lwork_min = (std::max)(blas_int(1), (std::max)(m,n));  // take into account requirements of geqrf() _and_ orgqr()/ungqr()
    blas_int k         = (std::min)(m,n);
    blas_int info      = 0;
    
    podarray<eT> tau( static_cast<uword>(k) );
    
    eT        work_query[2];
    blas_int lwork_query = -1;
    
    lapack::geqrf(&m, &n, Q.memptr(), &m, tau.memptr(), &work_query[0], &lwork_query, &info);
    
    if(info == 0)
      {
      const blas_int lwork_proposed = static_cast<blas_int>( access::tmp_real(work_query[0]) );
      
      lwork = (lwork_proposed > lwork_min) ? lwork_proposed : lwork_min;
      }
    else
      {
      return false;
      }
    
    podarray<eT> work( static_cast<uword>(lwork) );
    
    lapack::geqrf(&m, &n, Q.memptr(), &m, tau.memptr(), work.memptr(), &lwork, &info);
    
    // Q now has the elements on and above the diagonal of the array
    // contain the min(M,N)-by-N upper trapezoidal matrix Q
    // (Q is upper triangular if m >= n);
    // the elements below the diagonal, with the array TAU,
    // represent the orthogonal matrix Q as a product of min(m,n) elementary reflectors.
    
    R.set_size(Q_n_cols, Q_n_cols);
    
    //
    // construct R
    
    for(uword col=0; col < Q_n_cols; ++col)
      {
      for(uword row=0; row <= col; ++row)
        {
        R.at(row,col) = Q.at(row,col);
        }
      
      for(uword row=(col+1); row < Q_n_cols; ++row)
        {
        R.at(row,col) = eT(0);
        }
      }
    
    if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )
      {
      lapack::orgqr(&m, &n, &k, Q.memptr(), &m, tau.memptr(), work.memptr(), &lwork, &info);
      }
    else
    if( (is_supported_complex_float<eT>::value == true) || (is_supported_complex_double<eT>::value == true) )
      {
      lapack::ungqr(&m, &n, &k, Q.memptr(), &m, tau.memptr(), work.memptr(), &lwork, &info);
      }
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(Q);
    arma_ignore(R);
    arma_ignore(X);
    arma_stop("qr_econ(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename eT, typename T1>
inline
bool
auxlib::svd(Col<eT>& S, const Base<eT,T1>& X, uword& X_n_rows, uword& X_n_cols)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    Mat<eT> A(X.get_ref());
    
    X_n_rows = A.n_rows;
    X_n_cols = A.n_cols;
    
    if(A.is_empty())
      {
      S.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    Mat<eT> U(1, 1);
    Mat<eT> V(1, A.n_cols);
    
    char jobu  = 'N';
    char jobvt = 'N';
    
    blas_int m          = A.n_rows;
    blas_int n          = A.n_cols;
    blas_int min_mn     = (std::min)(m,n);
    blas_int lda        = A.n_rows;
    blas_int ldu        = U.n_rows;
    blas_int ldvt       = V.n_rows;
    blas_int lwork      = 0;
    blas_int lwork_min  = (std::max)( blas_int(1), (std::max)( (3*min_mn + (std::max)(m,n)), 5*min_mn ) );
    blas_int info   = 0;
    
    S.set_size( static_cast<uword>(min_mn) );
    
    eT        work_query[2];
    blas_int lwork_query = -1;
    
    lapack::gesvd<eT>
      (
      &jobu, &jobvt, &m, &n, A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, V.memptr(), &ldvt, &work_query[0], &lwork_query, &info
      );
    
    if(info == 0)
      {
      const blas_int lwork_proposed = static_cast<blas_int>( work_query[0] );
      
      lwork = (lwork_proposed > lwork_min) ? lwork_proposed : lwork_min;
      
      podarray<eT> work( static_cast<uword>(lwork) );
      
      lapack::gesvd<eT>
        (
        &jobu, &jobvt, &m, &n, A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, V.memptr(), &ldvt, work.memptr(), &lwork, &info
        );
      }
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(S);
    arma_ignore(X);
    arma_ignore(X_n_rows);
    arma_ignore(X_n_cols);
    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename T, typename T1>
inline
bool
auxlib::svd(Col<T>& S, const Base<std::complex<T>, T1>& X, uword& X_n_rows, uword& X_n_cols)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    typedef std::complex<T> eT;
    
    Mat<eT> A(X.get_ref());
    
    X_n_rows = A.n_rows;
    X_n_cols = A.n_cols;
    
    if(A.is_empty())
      {
      S.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    Mat<eT> U(1, 1);
    Mat<eT> V(1, A.n_cols);
    
    char jobu  = 'N';
    char jobvt = 'N';
    
    blas_int  m      = A.n_rows;
    blas_int  n      = A.n_cols;
    blas_int  min_mn = (std::min)(m,n);
    blas_int  lda    = A.n_rows;
    blas_int  ldu    = U.n_rows;
    blas_int  ldvt   = V.n_rows;
    blas_int  lwork  = 3 * ( (std::max)(blas_int(1), 2*min_mn+(std::max)(m,n) ) );
    blas_int  info   = 0;
    
    S.set_size( static_cast<uword>(min_mn) );
    
    podarray<eT>   work( static_cast<uword>(lwork   ) );
    podarray< T>  rwork( static_cast<uword>(5*min_mn) );
    
    // let gesvd_() calculate the optimum size of the workspace
    blas_int lwork_tmp = -1;
    
    lapack::cx_gesvd<T>
      (
      &jobu, &jobvt,
      &m, &n,
      A.memptr(), &lda,
      S.memptr(),
      U.memptr(), &ldu,
      V.memptr(), &ldvt,
      work.memptr(), &lwork_tmp,
      rwork.memptr(),
      &info
      );
    
    if(info == 0)
      {
      blas_int proposed_lwork = static_cast<blas_int>(real(work[0]));
      if(proposed_lwork > lwork)
        {
        lwork = proposed_lwork;
        work.set_size( static_cast<uword>(lwork) );
        }
      
      lapack::cx_gesvd<T>
        (
        &jobu, &jobvt,
        &m, &n,
        A.memptr(), &lda,
        S.memptr(),
        U.memptr(), &ldu,
        V.memptr(), &ldvt,
        work.memptr(), &lwork,
        rwork.memptr(),
        &info
        );
      }
        
    return (info == 0);
    }
  #else
    {
    arma_ignore(S);
    arma_ignore(X);
    arma_ignore(X_n_rows);
    arma_ignore(X_n_cols);

    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename eT, typename T1>
inline
bool
auxlib::svd(Col<eT>& S, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  uword junk;
  return auxlib::svd(S, X, junk, junk);
  }



template<typename T, typename T1>
inline
bool
auxlib::svd(Col<T>& S, const Base<std::complex<T>, T1>& X)
  {
  arma_extra_debug_sigprint();
  
  uword junk;
  return auxlib::svd(S, X, junk, junk);
  }



template<typename eT, typename T1>
inline
bool
auxlib::svd(Mat<eT>& U, Col<eT>& S, Mat<eT>& V, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    Mat<eT> A(X.get_ref());
    
    if(A.is_empty())
      {
      U.eye(A.n_rows, A.n_rows);
      S.reset();
      V.eye(A.n_cols, A.n_cols);
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    U.set_size(A.n_rows, A.n_rows);
    V.set_size(A.n_cols, A.n_cols);
    
    char jobu  = 'A';
    char jobvt = 'A';
    
    blas_int  m          = blas_int(A.n_rows);
    blas_int  n          = blas_int(A.n_cols);
    blas_int  min_mn     = (std::min)(m,n);
    blas_int  lda        = blas_int(A.n_rows);
    blas_int  ldu        = blas_int(U.n_rows);
    blas_int  ldvt       = blas_int(V.n_rows);
    blas_int  lwork_min  = (std::max)( blas_int(1), (std::max)( (3*min_mn + (std::max)(m,n)), 5*min_mn ) );
    blas_int  lwork      = 0;
    blas_int  info       = 0;
    
    S.set_size( static_cast<uword>(min_mn) );
    
    // let gesvd_() calculate the optimum size of the workspace
    eT        work_query[2];
    blas_int lwork_query = -1;
    
    lapack::gesvd<eT>
      (
      &jobu, &jobvt, &m, &n, A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, V.memptr(), &ldvt, &work_query[0], &lwork_query, &info
      );
    
    if(info == 0)
      {
      const blas_int lwork_proposed = static_cast<blas_int>( work_query[0] );
      
      lwork = (lwork_proposed > lwork_min) ? lwork_proposed : lwork_min;
      
      podarray<eT> work( static_cast<uword>(lwork) );
      
      lapack::gesvd<eT>
        (
        &jobu, &jobvt, &m, &n, A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, V.memptr(), &ldvt, work.memptr(), &lwork, &info
        );
      
      op_strans::apply_mat_inplace(V);
      }
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(U);
    arma_ignore(S);
    arma_ignore(V);
    arma_ignore(X);
    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename T, typename T1>
inline
bool
auxlib::svd(Mat< std::complex<T> >& U, Col<T>& S, Mat< std::complex<T> >& V, const Base< std::complex<T>, T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    typedef std::complex<T> eT;
    
    Mat<eT> A(X.get_ref());
    
    if(A.is_empty())
      {
      U.eye(A.n_rows, A.n_rows);
      S.reset();
      V.eye(A.n_cols, A.n_cols);
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    U.set_size(A.n_rows, A.n_rows);
    V.set_size(A.n_cols, A.n_cols);
    
    char jobu  = 'A';
    char jobvt = 'A';
    
    blas_int  m      = blas_int(A.n_rows);
    blas_int  n      = blas_int(A.n_cols);
    blas_int  min_mn = (std::min)(m,n);
    blas_int  lda    = blas_int(A.n_rows);
    blas_int  ldu    = blas_int(U.n_rows);
    blas_int  ldvt   = blas_int(V.n_rows);
    blas_int  lwork  = 3 * ( (std::max)(blas_int(1), 2*min_mn + (std::max)(m,n) ) );
    blas_int  info   = 0;
    
    S.set_size( static_cast<uword>(min_mn) );
    
    podarray<eT>  work( static_cast<uword>(lwork   ) );
    podarray<T>  rwork( static_cast<uword>(5*min_mn) );
    
    // let gesvd_() calculate the optimum size of the workspace
    blas_int lwork_tmp = -1;
    lapack::cx_gesvd<T>
     (
     &jobu, &jobvt,
     &m, &n,
     A.memptr(), &lda,
     S.memptr(),
     U.memptr(), &ldu,
     V.memptr(), &ldvt,
     work.memptr(), &lwork_tmp,
     rwork.memptr(),
     &info
     );
    
    if(info == 0)
      {
      blas_int proposed_lwork = static_cast<blas_int>(real(work[0]));
      
      if(proposed_lwork > lwork)
        {
        lwork = proposed_lwork;
        work.set_size( static_cast<uword>(lwork) );
        }
      
      lapack::cx_gesvd<T>
        (
        &jobu, &jobvt,
        &m, &n,
        A.memptr(), &lda,
        S.memptr(),
        U.memptr(), &ldu,
        V.memptr(), &ldvt,
        work.memptr(), &lwork,
        rwork.memptr(),
        &info
        );
      
      op_htrans::apply_mat_inplace(V);
      }
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(U);
    arma_ignore(S);
    arma_ignore(V);
    arma_ignore(X);
    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename eT, typename T1>
inline
bool
auxlib::svd_econ(Mat<eT>& U, Col<eT>& S, Mat<eT>& V, const Base<eT,T1>& X, const char mode)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    Mat<eT> A(X.get_ref());
    
    arma_debug_assert_blas_size(A);
    
    blas_int m      = blas_int(A.n_rows);
    blas_int n      = blas_int(A.n_cols);
    blas_int min_mn = (std::min)(m,n);
    blas_int lda    = blas_int(A.n_rows);
    
    S.set_size( static_cast<uword>(min_mn) );
    
    blas_int ldu  = 0;
    blas_int ldvt = 0;
    
    char jobu;
    char jobvt;
    
    switch(mode)
      {
      case 'l':
        jobu  = 'S';
        jobvt = 'N';
        
        ldu  = m;
        ldvt = 1;
        
        U.set_size( static_cast<uword>(ldu), static_cast<uword>(min_mn) );
        V.reset();
        
        break;
      
      
      case 'r':
        jobu  = 'N';
        jobvt = 'S';
        
        ldu = 1;
        ldvt = (std::min)(m,n);
        
        U.reset();
        V.set_size( static_cast<uword>(ldvt), static_cast<uword>(n) );
        
        break;
      
      
      case 'b':
        jobu  = 'S';
        jobvt = 'S';
        
        ldu  = m;
        ldvt = (std::min)(m,n);
        
        U.set_size( static_cast<uword>(ldu),  static_cast<uword>(min_mn) );
        V.set_size( static_cast<uword>(ldvt), static_cast<uword>(n     ) );
        
        break;
      
      
      default:
        U.reset();
        S.reset();
        V.reset();
        return false;
      }
    
    
    if(A.is_empty())
      {
      U.eye();
      S.reset();
      V.eye();
      return true;
      }
    
    
    blas_int lwork = 3 * ( (std::max)(blas_int(1), (std::max)( (3*min_mn + (std::max)(m,n)), 5*min_mn ) ) );
    blas_int info  = 0;
    
    
    podarray<eT> work( static_cast<uword>(lwork) );
    
    // let gesvd_() calculate the optimum size of the workspace
    blas_int lwork_tmp = -1;
    
    lapack::gesvd<eT>
      (
      &jobu, &jobvt,
      &m, &n,
      A.memptr(), &lda,
      S.memptr(),
      U.memptr(), &ldu,
      V.memptr(), &ldvt,
      work.memptr(), &lwork_tmp,
      &info
      );
    
    if(info == 0)
      {
      blas_int proposed_lwork = static_cast<blas_int>(work[0]);
      if(proposed_lwork > lwork)
        {
        lwork = proposed_lwork;
        work.set_size( static_cast<uword>(lwork) );
        }
      
      lapack::gesvd<eT>
        (
        &jobu, &jobvt,
        &m, &n,
        A.memptr(), &lda,
        S.memptr(),
        U.memptr(), &ldu,
        V.memptr(), &ldvt,
        work.memptr(), &lwork,
        &info
        );
      
      op_strans::apply_mat_inplace(V);
      }
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(U);
    arma_ignore(S);
    arma_ignore(V);
    arma_ignore(X);
    arma_ignore(mode);
    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename T, typename T1>
inline
bool
auxlib::svd_econ(Mat< std::complex<T> >& U, Col<T>& S, Mat< std::complex<T> >& V, const Base< std::complex<T>, T1>& X, const char mode)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    typedef std::complex<T> eT;
    
    Mat<eT> A(X.get_ref());
    
    arma_debug_assert_blas_size(A);
    
    blas_int m      = blas_int(A.n_rows);
    blas_int n      = blas_int(A.n_cols);
    blas_int min_mn = (std::min)(m,n);
    blas_int lda    = blas_int(A.n_rows);
    
    S.set_size( static_cast<uword>(min_mn) );
    
    blas_int ldu  = 0;
    blas_int ldvt = 0;
    
    char jobu;
    char jobvt;
    
    switch(mode)
      {
      case 'l':
        jobu  = 'S';
        jobvt = 'N';
        
        ldu  = m;
        ldvt = 1;
        
        U.set_size( static_cast<uword>(ldu), static_cast<uword>(min_mn) );
        V.reset();
        
        break;
      
      
      case 'r':
        jobu  = 'N';
        jobvt = 'S';
        
        ldu  = 1;
        ldvt = (std::min)(m,n);
        
        U.reset();
        V.set_size( static_cast<uword>(ldvt), static_cast<uword>(n) );
        
        break;
      
      
      case 'b':
        jobu  = 'S';
        jobvt = 'S';
        
        ldu  = m;
        ldvt = (std::min)(m,n);
        
        U.set_size( static_cast<uword>(ldu),  static_cast<uword>(min_mn) );
        V.set_size( static_cast<uword>(ldvt), static_cast<uword>(n)      );
        
        break;
      
      
      default:
        U.reset();
        S.reset();
        V.reset();
        return false;
      }
    
    
    if(A.is_empty())
      {
      U.eye();
      S.reset();
      V.eye();
      return true;
      }
    
    
    blas_int lwork = 3 * ( (std::max)(blas_int(1), (std::max)( (3*min_mn + (std::max)(m,n)), 5*min_mn ) ) );
    blas_int info  = 0;
    
    
    podarray<eT>  work( static_cast<uword>(lwork   ) );
    podarray<T>  rwork( static_cast<uword>(5*min_mn) );
    
    // let gesvd_() calculate the optimum size of the workspace
    blas_int lwork_tmp = -1;
    
    lapack::cx_gesvd<T>
      (
      &jobu, &jobvt,
      &m, &n,
      A.memptr(), &lda,
      S.memptr(),
      U.memptr(), &ldu,
      V.memptr(), &ldvt,
      work.memptr(), &lwork_tmp,
      rwork.memptr(),
      &info
      );
    
    if(info == 0)
      {
      blas_int proposed_lwork = static_cast<blas_int>(real(work[0]));
      if(proposed_lwork > lwork)
        {
        lwork = proposed_lwork;
        work.set_size( static_cast<uword>(lwork) );
        }
      
      lapack::cx_gesvd<T>
        (
        &jobu, &jobvt,
        &m, &n,
        A.memptr(), &lda,
        S.memptr(),
        U.memptr(), &ldu,
        V.memptr(), &ldvt,
        work.memptr(), &lwork,
        rwork.memptr(),
        &info
        );
      
      op_htrans::apply_mat_inplace(V);
      }
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(U);
    arma_ignore(S);
    arma_ignore(V);
    arma_ignore(X);
    arma_ignore(mode);
    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



// EXPERIMENTAL
template<typename eT, typename T1>
inline
bool
auxlib::svd_dc(Col<eT>& S, const Base<eT,T1>& X, uword& X_n_rows, uword& X_n_cols)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    Mat<eT> A(X.get_ref());
    
    X_n_rows = A.n_rows;
    X_n_cols = A.n_cols;
    
    if(A.is_empty())
      {
      S.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    Mat<eT> U(1, 1);
    Mat<eT> V(1, 1);
    
    char jobz = 'N';
    
    blas_int  m      = blas_int(A.n_rows);
    blas_int  n      = blas_int(A.n_cols);
    blas_int  min_mn = (std::min)(m,n);
    blas_int  lda    = blas_int(A.n_rows);
    blas_int  ldu    = blas_int(U.n_rows);
    blas_int  ldvt   = blas_int(V.n_rows);
    blas_int  lwork  = 3 * ( 3*min_mn + std::max( std::max(m,n), 7*min_mn ) );
    blas_int  info   = 0;
    
    S.set_size( static_cast<uword>(min_mn) );
    
    podarray<eT>        work( static_cast<uword>(lwork   ) );
    podarray<blas_int> iwork( static_cast<uword>(8*min_mn) );
    
    lapack::gesdd<eT>
      (
      &jobz, &m, &n,
      A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, V.memptr(), &ldvt,
      work.memptr(), &lwork, iwork.memptr(), &info
      );
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(S);
    arma_ignore(X);
    arma_ignore(X_n_rows);
    arma_ignore(X_n_cols);
    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



// EXPERIMENTAL
template<typename T, typename T1>
inline
bool
auxlib::svd_dc(Col<T>& S, const Base<std::complex<T>, T1>& X, uword& X_n_rows, uword& X_n_cols)
  {
  arma_extra_debug_sigprint();
  
  #if (defined(ARMA_USE_LAPACK) && defined(ARMA_DONT_USE_CX_GESDD))
    {
    arma_extra_debug_print("auxlib::svd_dc(): redirecting to auxlib::svd(), as use of lapack::cx_gesdd() is disabled");
    
    return auxlib::svd(S, X, X_n_rows, X_n_cols);
    }
  #elif defined(ARMA_USE_LAPACK)
    {
    typedef std::complex<T> eT;
    
    Mat<eT> A(X.get_ref());
    
    X_n_rows = A.n_rows;
    X_n_cols = A.n_cols;
    
    if(A.is_empty())
      {
      S.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    Mat<eT> U(1, 1);
    Mat<eT> V(1, 1);
    
    char jobz = 'N';
    
    blas_int  m      = blas_int(A.n_rows);
    blas_int  n      = blas_int(A.n_cols);
    blas_int  min_mn = (std::min)(m,n);
    blas_int  lda    = blas_int(A.n_rows);
    blas_int  ldu    = blas_int(U.n_rows);
    blas_int  ldvt   = blas_int(V.n_rows);
    blas_int  lwork  = 3 * (2*min_mn + std::max(m,n));  
    blas_int  info   = 0;
    
    S.set_size( static_cast<uword>(min_mn) );
    
    podarray<eT>        work( static_cast<uword>(lwork   ) );
    podarray<T>        rwork( static_cast<uword>(7*min_mn) );  // LAPACK 3.4.2 docs state 5*min(m,n), while zgesdd() seems to write past the end 
    podarray<blas_int> iwork( static_cast<uword>(8*min_mn) );
    
    lapack::cx_gesdd<T>
      (
      &jobz, &m, &n,
      A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, V.memptr(), &ldvt,
      work.memptr(), &lwork, rwork.memptr(), iwork.memptr(), &info
      );
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(S);
    arma_ignore(X);
    arma_ignore(X_n_rows);
    arma_ignore(X_n_cols);
    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



// EXPERIMENTAL
template<typename eT, typename T1>
inline
bool
auxlib::svd_dc(Col<eT>& S, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  uword junk;
  return auxlib::svd_dc(S, X, junk, junk);
  }



// EXPERIMENTAL
template<typename T, typename T1>
inline
bool
auxlib::svd_dc(Col<T>& S, const Base<std::complex<T>, T1>& X)
  {
  arma_extra_debug_sigprint();
  
  uword junk;
  return auxlib::svd_dc(S, X, junk, junk);
  }



template<typename eT, typename T1>
inline
bool
auxlib::svd_dc(Mat<eT>& U, Col<eT>& S, Mat<eT>& V, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    Mat<eT> A(X.get_ref());
    
    if(A.is_empty())
      {
      U.eye(A.n_rows, A.n_rows);
      S.reset();
      V.eye(A.n_cols, A.n_cols);
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    U.set_size(A.n_rows, A.n_rows);
    V.set_size(A.n_cols, A.n_cols);
    
    char jobz = 'A';
    
    blas_int  m      = blas_int(A.n_rows);
    blas_int  n      = blas_int(A.n_cols);
    blas_int  min_mn = (std::min)(m,n);
    blas_int  max_mn = (std::max)(m,n);
    blas_int  lda    = blas_int(A.n_rows);
    blas_int  ldu    = blas_int(U.n_rows);
    blas_int  ldvt   = blas_int(V.n_rows);
    blas_int  lwork1 = 3*min_mn*min_mn + (std::max)( max_mn, 4*min_mn*min_mn + 4*min_mn          );
    blas_int  lwork2 = 3*min_mn        + (std::max)( max_mn, 4*min_mn*min_mn + 3*min_mn + max_mn );
    blas_int  lwork  = 2 * ((std::max)(lwork1, lwork2));  // due to differences between lapack 3.1 and 3.4
    blas_int  info   = 0;
    
    S.set_size( static_cast<uword>(min_mn) );
    
    podarray<eT>        work( static_cast<uword>(lwork   ) );
    podarray<blas_int> iwork( static_cast<uword>(8*min_mn) );
    
    lapack::gesdd<eT>
      (
      &jobz, &m, &n,
      A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, V.memptr(), &ldvt,
      work.memptr(), &lwork, iwork.memptr(), &info
      );
    
    op_strans::apply_mat_inplace(V);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(U);
    arma_ignore(S);
    arma_ignore(V);
    arma_ignore(X);
    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename T, typename T1>
inline
bool
auxlib::svd_dc(Mat< std::complex<T> >& U, Col<T>& S, Mat< std::complex<T> >& V, const Base< std::complex<T>, T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if (defined(ARMA_USE_LAPACK) && defined(ARMA_DONT_USE_CX_GESDD))
    {
    arma_extra_debug_print("auxlib::svd_dc(): redirecting to auxlib::svd(), as use of lapack::cx_gesdd() is disabled");
    
    return auxlib::svd(U, S, V, X);
    }
  #elif defined(ARMA_USE_LAPACK)
    {
    typedef std::complex<T> eT;
    
    Mat<eT> A(X.get_ref());
    
    if(A.is_empty())
      {
      U.eye(A.n_rows, A.n_rows);
      S.reset();
      V.eye(A.n_cols, A.n_cols);
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    U.set_size(A.n_rows, A.n_rows);
    V.set_size(A.n_cols, A.n_cols);
    
    char jobz = 'A';
    
    blas_int m       = blas_int(A.n_rows);
    blas_int n       = blas_int(A.n_cols);
    blas_int min_mn  = (std::min)(m,n);
    blas_int max_mn  = (std::max)(m,n);
    blas_int lda     = blas_int(A.n_rows);
    blas_int ldu     = blas_int(U.n_rows);
    blas_int ldvt    = blas_int(V.n_rows);
    blas_int lwork   = 2 * (min_mn*min_mn + 2*min_mn + max_mn);
    blas_int lrwork1 = 5*min_mn*min_mn + 7*min_mn;
    blas_int lrwork2 = min_mn * ((std::max)(5*min_mn+7, 2*max_mn + 2*min_mn+1));
    blas_int lrwork  = (std::max)(lrwork1, lrwork2);  // due to differences between lapack 3.1 and 3.4
    blas_int info    = 0;
    
    S.set_size( static_cast<uword>(min_mn) );
    
    podarray<eT>        work( static_cast<uword>(lwork   ) );
    podarray<T>        rwork( static_cast<uword>(lrwork  ) );
    podarray<blas_int> iwork( static_cast<uword>(8*min_mn) );
    
    lapack::cx_gesdd<T>
      (
      &jobz, &m, &n,
      A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, V.memptr(), &ldvt,
      work.memptr(), &lwork, rwork.memptr(), iwork.memptr(), &info
      );
    
    op_htrans::apply_mat_inplace(V);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(U);
    arma_ignore(S);
    arma_ignore(V);
    arma_ignore(X);
    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename eT, typename T1>
inline
bool
auxlib::svd_dc_econ(Mat<eT>& U, Col<eT>& S, Mat<eT>& V, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    Mat<eT> A(X.get_ref());
    
    arma_debug_assert_blas_size(A);
    
    char jobz = 'S';
    
    blas_int m      = blas_int(A.n_rows);
    blas_int n      = blas_int(A.n_cols);
    blas_int min_mn = (std::min)(m,n);
    blas_int max_mn = (std::max)(m,n);
    blas_int lda    = blas_int(A.n_rows);
    blas_int ldu    = m;
    blas_int ldvt   = min_mn;
    blas_int lwork1 = 3*min_mn*min_mn + (std::max)( max_mn, 4*min_mn*min_mn + 4*min_mn          );
    blas_int lwork2 = 3*min_mn        + (std::max)( max_mn, 4*min_mn*min_mn + 3*min_mn + max_mn );
    blas_int lwork  = 2 * ((std::max)(lwork1, lwork2));  // due to differences between lapack 3.1 and 3.4
    blas_int info   = 0;
    
    if(A.is_empty())
      {
      U.eye();
      S.reset();
      V.eye( static_cast<uword>(n), static_cast<uword>(min_mn) );
      return true;
      }
    
    S.set_size( static_cast<uword>(min_mn) );
    
    U.set_size( static_cast<uword>(m), static_cast<uword>(min_mn) );
    
    V.set_size( static_cast<uword>(min_mn), static_cast<uword>(n) );
    
    podarray<eT>        work( static_cast<uword>(lwork   ) );
    podarray<blas_int> iwork( static_cast<uword>(8*min_mn) );
    
    lapack::gesdd<eT>
      (
      &jobz, &m, &n,
      A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, V.memptr(), &ldvt,
      work.memptr(), &lwork, iwork.memptr(), &info
      );
    
    op_strans::apply_mat_inplace(V);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(U);
    arma_ignore(S);
    arma_ignore(V);
    arma_ignore(X);
    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename T, typename T1>
inline
bool
auxlib::svd_dc_econ(Mat< std::complex<T> >& U, Col<T>& S, Mat< std::complex<T> >& V, const Base< std::complex<T>, T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if (defined(ARMA_USE_LAPACK) && defined(ARMA_DONT_USE_CX_GESDD))
    {
    arma_extra_debug_print("auxlib::svd_dc_econ(): redirecting to auxlib::svd_econ(), as use of lapack::cx_gesdd() is disabled");
    
    return auxlib::svd_econ(U, S, V, X, 'b');
    }
  #elif defined(ARMA_USE_LAPACK)
    {
    typedef std::complex<T> eT;
    
    Mat<eT> A(X.get_ref());
    
    arma_debug_assert_blas_size(A);
    
    char jobz = 'S';
    
    blas_int m       = blas_int(A.n_rows);
    blas_int n       = blas_int(A.n_cols);
    blas_int min_mn  = (std::min)(m,n);
    blas_int max_mn  = (std::max)(m,n);
    blas_int lda     = blas_int(A.n_rows);
    blas_int ldu     = m;
    blas_int ldvt    = min_mn;
    blas_int lwork   = 2 * (min_mn*min_mn + 2*min_mn + max_mn);
    blas_int lrwork1 = 5*min_mn*min_mn + 7*min_mn;
    blas_int lrwork2 = min_mn * ((std::max)(5*min_mn+7, 2*max_mn + 2*min_mn+1));
    blas_int lrwork  = (std::max)(lrwork1, lrwork2);  // due to differences between lapack 3.1 and 3.4
    blas_int info    = 0;
    
    if(A.is_empty())
      {
      U.eye();
      S.reset();
      V.eye( static_cast<uword>(n), static_cast<uword>(min_mn) );
      return true;
      }
    
    S.set_size( static_cast<uword>(min_mn) );
    
    U.set_size( static_cast<uword>(m), static_cast<uword>(min_mn) );
    
    V.set_size( static_cast<uword>(min_mn), static_cast<uword>(n) );
    
    podarray<eT>        work( static_cast<uword>(lwork   ) );
    podarray<T>        rwork( static_cast<uword>(lrwork  ) );
    podarray<blas_int> iwork( static_cast<uword>(8*min_mn) );
    
    lapack::cx_gesdd<T>
      (
      &jobz, &m, &n,
      A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, V.memptr(), &ldvt,
      work.memptr(), &lwork, rwork.memptr(), iwork.memptr(), &info
      );
    
    op_htrans::apply_mat_inplace(V);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(U);
    arma_ignore(S);
    arma_ignore(V);
    arma_ignore(X);
    arma_stop("svd(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//! Solve a system of linear equations.
//! Assumes that A.n_rows = A.n_cols and B.n_rows = A.n_rows
template<typename eT, typename T1>
inline
bool
auxlib::solve(Mat<eT>& out, Mat<eT>& A, const Base<eT,T1>& X, const bool slow)
  {
  arma_extra_debug_sigprint();
  
  bool status = false;
  
  const uword A_n_rows = A.n_rows;
  
  if( (A_n_rows <= 4) && (slow == false) )
    {
    Mat<eT> A_inv(A_n_rows, A_n_rows);
    
    status = auxlib::inv_noalias_tinymat(A_inv, A, A_n_rows);
    
    if(status == true)
      {
      const unwrap_check<T1> Y( X.get_ref(), out );
      const Mat<eT>& B     = Y.M;
      
      const uword B_n_rows = B.n_rows;
      const uword B_n_cols = B.n_cols;
      
      arma_debug_check( (A_n_rows != B_n_rows), "solve(): number of rows in the given objects must be the same" );
      
      if(A.is_empty() || B.is_empty())
        {
        out.zeros(A.n_cols, B_n_cols);
        return true;
        }
      
      out.set_size(A_n_rows, B_n_cols);
      
      gemm_emul<false,false,false,false>::apply(out, A_inv, B);
      
      return true;
      }
    }
  
  if( (A_n_rows > 4) || (status == false) )
    {
    out = X.get_ref();
    
    const uword B_n_rows = out.n_rows;
    const uword B_n_cols = out.n_cols;
      
    arma_debug_check( (A_n_rows != B_n_rows), "solve(): number of rows in the given objects must be the same" );
      
    if(A.is_empty() || out.is_empty())
      {
      out.zeros(A.n_cols, B_n_cols);
      return true;
      }
    
    #if defined(ARMA_USE_ATLAS)
      {
      arma_debug_assert_atlas_size(A);
      
      podarray<int> ipiv(A_n_rows + 2);  // +2 for paranoia: old versions of Atlas might be trashing memory
      
      int info = atlas::clapack_gesv<eT>(atlas::CblasColMajor, A_n_rows, B_n_cols, A.memptr(), A_n_rows, ipiv.memptr(), out.memptr(), A_n_rows);
      
      return (info == 0);
      }
    #elif defined(ARMA_USE_LAPACK)
      {
      arma_debug_assert_blas_size(A);
      
      blas_int n    = blas_int(A_n_rows);  // assuming A is square
      blas_int lda  = blas_int(A_n_rows);
      blas_int ldb  = blas_int(A_n_rows);
      blas_int nrhs = blas_int(B_n_cols);
      blas_int info = 0;
      
      podarray<blas_int> ipiv(A_n_rows + 2);  // +2 for paranoia: some versions of Lapack might be trashing memory
      
      arma_extra_debug_print("lapack::gesv()");
      lapack::gesv<eT>(&n, &nrhs, A.memptr(), &lda, ipiv.memptr(), out.memptr(), &ldb, &info);
      
      arma_extra_debug_print("lapack::gesv() -- finished");
      
      return (info == 0);
      }
    #else
      {
      arma_stop("solve(): use of ATLAS or LAPACK needs to be enabled");
      return false;
      }
    #endif
    }
  
  return true;
  }



//! Solve an over-determined system.
//! Assumes that A.n_rows > A.n_cols and B.n_rows = A.n_rows
template<typename eT, typename T1>
inline
bool
auxlib::solve_od(Mat<eT>& out, Mat<eT>& A, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    Mat<eT> tmp = X.get_ref();
    
    const uword A_n_rows = A.n_rows;
    const uword A_n_cols = A.n_cols;
    
    const uword B_n_rows = tmp.n_rows;
    const uword B_n_cols = tmp.n_cols;
      
    arma_debug_check( (A_n_rows != B_n_rows), "solve(): number of rows in the given objects must be the same" );
    
    out.set_size(A_n_cols, B_n_cols);
    
    if(A.is_empty() || tmp.is_empty())
      {
      out.zeros();
      return true;
      }
    
    arma_debug_assert_blas_size(A,tmp);
    
    char trans = 'N';
    
    blas_int  m     = blas_int(A_n_rows);
    blas_int  n     = blas_int(A_n_cols);
    blas_int  lda   = blas_int(A_n_rows);
    blas_int  ldb   = blas_int(A_n_rows);
    blas_int  nrhs  = blas_int(B_n_cols);
    blas_int  lwork = 3 * ( (std::max)(blas_int(1), n + (std::max)(n, nrhs)) );
    blas_int  info  = 0;
    
    podarray<eT> work( static_cast<uword>(lwork) );
    
    // NOTE: the dgels() function in the lapack library supplied by ATLAS 3.6 seems to have problems
    arma_extra_debug_print("lapack::gels()");
    lapack::gels<eT>( &trans, &m, &n, &nrhs, A.memptr(), &lda, tmp.memptr(), &ldb, work.memptr(), &lwork, &info );
    
    arma_extra_debug_print("lapack::gels() -- finished");
    
    for(uword col=0; col<B_n_cols; ++col)
      {
      arrayops::copy( out.colptr(col), tmp.colptr(col), A_n_cols );
      }
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(out);
    arma_ignore(A);
    arma_ignore(X);
    arma_stop("solve(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//! Solve an under-determined system.
//! Assumes that A.n_rows < A.n_cols and B.n_rows = A.n_rows
template<typename eT, typename T1>
inline
bool
auxlib::solve_ud(Mat<eT>& out, Mat<eT>& A, const Base<eT,T1>& X)
  {
  arma_extra_debug_sigprint();
  
  // TODO: this function provides the same results as Octave 3.4.2.
  // TODO: however, these results are different than Matlab 7.12.0.635.
  // TODO: figure out whether both Octave and Matlab are correct, or only one of them
  
  #if defined(ARMA_USE_LAPACK)
    {
    const unwrap<T1>   Y( X.get_ref() );
    const Mat<eT>& B = Y.M;
    
    const uword A_n_rows = A.n_rows;
    const uword A_n_cols = A.n_cols;
    
    const uword B_n_rows = B.n_rows;
    const uword B_n_cols = B.n_cols;
    
    arma_debug_check( (A_n_rows != B_n_rows), "solve(): number of rows in the given objects must be the same" );
    
    // B could be an alias of "out", hence we need to check whether B is empty before setting the size of "out"
    if(A.is_empty() || B.is_empty())
      {
      out.zeros(A_n_cols, B_n_cols);
      return true;
      }
    
    arma_debug_assert_blas_size(A,B);
    
    char trans = 'N';
    
    blas_int  m     = blas_int(A_n_rows);
    blas_int  n     = blas_int(A_n_cols);
    blas_int  lda   = blas_int(A_n_rows);
    blas_int  ldb   = blas_int(A_n_cols);
    blas_int  nrhs  = blas_int(B_n_cols);
    blas_int  lwork = 3 * ( (std::max)(blas_int(1), m + (std::max)(m,nrhs)) );
    blas_int  info  = 0;
    
    Mat<eT> tmp(A_n_cols, B_n_cols);
    tmp.zeros();
    
    for(uword col=0; col<B_n_cols; ++col)
      {
      eT* tmp_colmem = tmp.colptr(col);
      
      arrayops::copy( tmp_colmem, B.colptr(col), B_n_rows );
      
      for(uword row=B_n_rows; row<A_n_cols; ++row)
        {
        tmp_colmem[row] = eT(0);
        }
      }
    
    podarray<eT> work( static_cast<uword>(lwork) );
    
    // NOTE: the dgels() function in the lapack library supplied by ATLAS 3.6 seems to have problems
    arma_extra_debug_print("lapack::gels()");
    lapack::gels<eT>( &trans, &m, &n, &nrhs, A.memptr(), &lda, tmp.memptr(), &ldb, work.memptr(), &lwork, &info );
    
    arma_extra_debug_print("lapack::gels() -- finished");
    
    out.set_size(A_n_cols, B_n_cols);
    
    for(uword col=0; col<B_n_cols; ++col)
      {
      arrayops::copy( out.colptr(col), tmp.colptr(col), A_n_cols );
      }
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(out);
    arma_ignore(A);
    arma_ignore(X);
    arma_stop("solve(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//
// solve_tr

template<typename eT>
inline
bool
auxlib::solve_tr(Mat<eT>& out, const Mat<eT>& A, const Mat<eT>& B, const uword layout)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    if(A.is_empty() || B.is_empty())
      {
      out.zeros(A.n_cols, B.n_cols);
      return true;
      }
    
    arma_debug_assert_blas_size(A,B);
    
    out = B;
    
    char     uplo  = (layout == 0) ? 'U' : 'L';
    char     trans = 'N';
    char     diag  = 'N';
    blas_int n     = blas_int(A.n_rows);
    blas_int nrhs  = blas_int(B.n_cols);
    blas_int info  = 0;
    
    lapack::trtrs<eT>(&uplo, &trans, &diag, &n, &nrhs, A.memptr(), &n, out.memptr(), &n, &info);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(out);
    arma_ignore(A);
    arma_ignore(B);
    arma_ignore(layout);
    arma_stop("solve(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//
// Schur decomposition

template<typename eT>
inline
bool
auxlib::schur_dec(Mat<eT>& Z, Mat<eT>& T, const Mat<eT>& A)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    arma_debug_check( (A.is_square() == false), "schur_dec(): given matrix is not square" );
    
    if(A.is_empty())
      {
      Z.reset();
      T.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    const uword A_n_rows = A.n_rows;
    
    Z.set_size(A_n_rows, A_n_rows);
    T = A;
    
    char    jobvs    = 'V';                // get Schur vectors (Z)
    char     sort    = 'N';                // do not sort eigenvalues/vectors
    blas_int* select = 0;                  // pointer to sorting function
    blas_int    n    = blas_int(A_n_rows);
    blas_int sdim    = 0;                  // output for sorting
    blas_int lwork   = 3 * ( (std::max)(blas_int(1), 3*n) );
    blas_int info    = 0;
    
    podarray<eT>       work( static_cast<uword>(lwork) );
    podarray<blas_int> bwork(A_n_rows);
    
    podarray<eT> wr(A_n_rows);             // output for eigenvalues
    podarray<eT> wi(A_n_rows);             // output for eigenvalues
    
    lapack::gees(&jobvs, &sort, select, &n, T.memptr(), &n, &sdim, wr.memptr(), wi.memptr(), Z.memptr(), &n, work.memptr(), &lwork, bwork.memptr(), &info);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(Z);
    arma_ignore(T);
    arma_ignore(A);
    
    arma_stop("schur_dec(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



template<typename cT>
inline
bool
auxlib::schur_dec(Mat<std::complex<cT> >& Z, Mat<std::complex<cT> >& T, const Mat<std::complex<cT> >& A)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_LAPACK)
    {
    arma_debug_check( (A.is_square() == false), "schur_dec(): matrix A is not square" );
    
    if(A.is_empty())
      {
      Z.reset();
      T.reset();
      return true;
      }
    
    arma_debug_assert_blas_size(A);
    
    typedef std::complex<cT> eT;
    
    const uword A_n_rows = A.n_rows;
    
    Z.set_size(A_n_rows, A_n_rows);
    T = A;
    
    char        jobvs = 'V';                // get Schur vectors (Z)
    char         sort = 'N';                // do not sort eigenvalues/vectors
    blas_int*  select = 0;                  // pointer to sorting function
    blas_int        n = blas_int(A_n_rows);
    blas_int     sdim = 0;                  // output for sorting
    blas_int lwork    = 3 * ( (std::max)(blas_int(1), 2*n) );
    blas_int info     = 0;
    
    podarray<eT>       work( static_cast<uword>(lwork) );
    podarray<blas_int> bwork(A_n_rows);
    
    podarray<eT>     w(A_n_rows);           // output for eigenvalues
    podarray<cT> rwork(A_n_rows);
    
    lapack::cx_gees(&jobvs, &sort, select, &n, T.memptr(), &n, &sdim, w.memptr(), Z.memptr(), &n, work.memptr(), &lwork, rwork.memptr(), bwork.memptr(), &info);
    
    return (info == 0);
    }
  #else
    {
    arma_ignore(Z);
    arma_ignore(T);
    arma_ignore(A);
    
    arma_stop("schur_dec(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }



//
// syl (solution of the Sylvester equation AX + XB = C)

template<typename eT>
inline
bool
auxlib::syl(Mat<eT>& X, const Mat<eT>& A, const Mat<eT>& B, const Mat<eT>& C)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check
    (
    (A.is_square() == false) || (B.is_square() == false),
    "syl(): given matrix is not square"
    );
    
  arma_debug_check
    (
    (C.n_rows != A.n_rows) || (C.n_cols != B.n_cols),
    "syl(): matrices are not conformant"
    );
  
  if(A.is_empty() || B.is_empty() || C.is_empty())
    {
    X.reset();
    return true;
    }
  
  #if defined(ARMA_USE_LAPACK)
    {
    Mat<eT> Z1, Z2, T1, T2;
    
    const bool status_sd1 = auxlib::schur_dec(Z1, T1, A);
    const bool status_sd2 = auxlib::schur_dec(Z2, T2, B);
    
    if( (status_sd1 == false) || (status_sd2 == false) )
      {
      return false;
      }
    
    char     trana = 'N';
    char     tranb = 'N';
    blas_int  isgn = +1;
    blas_int     m = blas_int(T1.n_rows);
    blas_int     n = blas_int(T2.n_cols);
    
    eT       scale = eT(0);
    blas_int  info = 0;
    
    Mat<eT> Y = trans(Z1) * C * Z2;
    
    lapack::trsyl<eT>(&trana, &tranb, &isgn, &m, &n, T1.memptr(), &m, T2.memptr(), &n, Y.memptr(), &m, &scale, &info);
    
    //Y /= scale;
    Y /= (-scale);
    
    X = Z1 * Y * trans(Z2);
    
    return (info >= 0);
    }
  #else
    {
    arma_stop("syl(): use of LAPACK needs to be enabled");
    return false;
    }
  #endif
  }
  
  
  
//
// lyap (solution of the continuous Lyapunov equation AX + XA^H + Q = 0)

template<typename eT>
inline
bool
auxlib::lyap(Mat<eT>& X, const Mat<eT>& A, const Mat<eT>& Q)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (A.is_square() == false), "lyap(): matrix A is not square");
  arma_debug_check( (Q.is_square() == false), "lyap(): matrix Q is not square");
  arma_debug_check( (A.n_rows != Q.n_rows),   "lyap(): matrices A and Q have different dimensions");
  
  Mat<eT> htransA;
  op_htrans::apply_mat_noalias(htransA, A);
  
  const Mat<eT> mQ = -Q;
  
  return auxlib::syl(X, A, htransA, mQ);
  }



//
// dlyap (solution of the discrete Lyapunov equation AXA^H - X + Q = 0)

template<typename eT>
inline
bool
auxlib::dlyap(Mat<eT>& X, const Mat<eT>& A, const Mat<eT>& Q)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (A.is_square() == false), "dlyap(): matrix A is not square");
  arma_debug_check( (Q.is_square() == false), "dlyap(): matrix Q is not square");
  arma_debug_check( (A.n_rows != Q.n_rows),   "dlyap(): matrices A and Q have different dimensions");
  
  const Col<eT> vecQ = reshape(Q, Q.n_elem, 1);
  
  const Mat<eT> M = eye< Mat<eT> >(Q.n_elem, Q.n_elem) - kron(conj(A), A);
  
  Col<eT> vecX;
  
  const bool status = solve(vecX, M, vecQ);
  
  if(status == true)
    {
    X = reshape(vecX, Q.n_rows, Q.n_cols);
    return true;
    }
  else
    {
    X.reset();
    return false;
    }
  }



//! @}
