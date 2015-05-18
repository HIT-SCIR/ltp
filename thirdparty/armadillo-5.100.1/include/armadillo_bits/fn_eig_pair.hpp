// Copyright (C) 2013 Conrad Sanderson
// Copyright (C) 2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_eig_pair
//! @{


//! eigenvalues for pair of N-by-N general real matrices (A,B)
template<typename T, typename T1>
inline
Col< std::complex<T> >
eig_pair
  (
  const Base<T, T1>& A,
  const Base<T, T1>& B,
  const typename arma_blas_type_only<T>::result* junk1 = 0,
  const typename         arma_not_cx<T>::result* junk2 = 0 
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  Mat<T> l_eigvec;
  Mat<T> r_eigvec;
  
  Col< std::complex<T> > eigval;
  
  const bool status = auxlib::eig_pair(eigval, l_eigvec, r_eigvec, A, B, 'n');
  
  if(status == false)
    {
    eigval.reset();
    arma_bad("eig_pair(): failed to converge");
    }
  
  return eigval;
  }



//! eigenvalues for pair of N-by-N general complex matrices (A,B)
template<typename T, typename T1>
inline
Col< std::complex<T> >
eig_pair
  (
  const Base< std::complex<T>, T1>& A, 
  const Base< std::complex<T>, T1>& B, 
  const typename arma_blas_type_only< std::complex<T> >::result* junk1 = 0,
  const typename        arma_cx_only< std::complex<T> >::result* junk2 = 0 
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  Mat< std::complex<T> > l_eigvec;
  Mat< std::complex<T> > r_eigvec;
  
  Col< std::complex<T> > eigval;
  
  const bool status = auxlib::eig_pair(eigval, l_eigvec, r_eigvec, A, B, 'n');
  
  if(status == false)
    {
    eigval.reset();
    arma_bad("eig_pair(): failed to converge");
    }
  
  return eigval;
  }



//! eigenvalues for pair of N-by-N general real matrices (A,B)
template<typename eT, typename T1, typename T2>
inline
bool
eig_pair
  (
         Col< std::complex<eT> >& eigval, 
  const Base< eT, T1 >&           A,
  const Base< eT, T2 >&           B,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  Mat<eT> l_eigvec;
  Mat<eT> r_eigvec;
  
  const bool status = auxlib::eig_pair(eigval, l_eigvec, r_eigvec, A, B, 'n');
  
  if(status == false)
    {
    eigval.reset();
    arma_bad("eig_pair(): failed to converge", false);
    }
  
  return status;
  }



//! eigenvalues for pair of N-by-N general complex matrices (A,B)
template<typename T, typename T1, typename T2>
inline
bool
eig_pair
  (
         Col< std::complex<T> >&     eigval, 
  const Base< std::complex<T>, T1 >& A,
  const Base< std::complex<T>, T2 >& B,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  Mat< std::complex<T> > l_eigvec;
  Mat< std::complex<T> > r_eigvec;
  
  const bool status = auxlib::eig_pair(eigval, l_eigvec, r_eigvec, A, B, 'n');
  
  if(status == false)
    {
    eigval.reset();
    arma_bad("eig_pair(): failed to converge", false);
    }
  
  return status;
  }



//! eigenvalues and eigenvectors for pair of N-by-N general real matrices (A,B)
template<typename eT, typename T1, typename T2>
inline
bool
eig_pair
  (
         Col< std::complex<eT> >& eigval, 
         Mat< std::complex<eT> >& eigvec,
  const Base< eT, T1 >&           A,
  const Base< eT, T2 >&           B,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  arma_debug_check( ( ((void*)(&eigval)) == ((void*)(&eigvec)) ), "eig_pair(): eigval is an alias of eigvec" );
  
  Mat<eT> dummy_eigvec;
  Mat<eT> tmp_eigvec;
  
  const bool status = auxlib::eig_pair(eigval, dummy_eigvec, tmp_eigvec, A, B, 'r');
  
  if(status == false)
    {
    eigval.reset();
    eigvec.reset();
    arma_bad("eig_pair(): failed to converge", false);
    }
  else
    {
    const uword n = eigval.n_elem;
    
    if(n > 0)
      {
      eigvec.set_size(n,n);
      
      // from LAPACK docs:
      // If the j-th and (j+1)-th eigenvalues form a complex conjugate pair, then
      // v(j) = VR(:,j)+i*VR(:,j+1) and v(j+1) = VR(:,j)-i*VR(:,j+1).
      
      for(uword j=0; j<n; ++j)
        {
        if( (j < n-1) && (eigval[j] == std::conj(eigval[j+1])) )
          {
          // eigvec.col(j)   = Mat< std::complex<eT> >( tmp_eigvec.col(j),  tmp_eigvec.col(j+1) );
          // eigvec.col(j+1) = Mat< std::complex<eT> >( tmp_eigvec.col(j), -tmp_eigvec.col(j+1) );
          
          for(uword i=0; i<n; ++i)
            {
            eigvec.at(i,j)   = std::complex<eT>( tmp_eigvec.at(i,j),  tmp_eigvec.at(i,j+1) );
            eigvec.at(i,j+1) = std::complex<eT>( tmp_eigvec.at(i,j), -tmp_eigvec.at(i,j+1) );
            }
          
          ++j;
          }
        else
          {
          // eigvec.col(i) = tmp_eigvec.col(i);
          
          for(uword i=0; i<n; ++i)
            {
            eigvec.at(i,j) = std::complex<eT>(tmp_eigvec.at(i,j), eT(0));
            }
          }
        }
      }
    }
  
  return status;
  }



//! eigenvalues and eigenvectors for pair of N-by-N general complex matrices (A,B)
template<typename T, typename T1, typename T2>
inline
bool
eig_pair
  (
         Col< std::complex<T> >&     eigval, 
         Mat< std::complex<T> >&     eigvec,
  const Base< std::complex<T>, T1 >& A,
  const Base< std::complex<T>, T2 >& B,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  arma_debug_check( ( ((void*)(&eigval)) == ((void*)(&eigvec)) ), "eig_pair(): eigval is an alias of eigvec" );
  
  Mat< std::complex<T> > dummy_eigvec;
  
  const bool status = auxlib::eig_pair(eigval, dummy_eigvec, eigvec, A, B, 'r');
  
  if(status == false)
    {
    eigval.reset();
    eigvec.reset();
    arma_bad("eig_pair(): failed to converge", false);
    }
  
  return status;
  }


//! @}
