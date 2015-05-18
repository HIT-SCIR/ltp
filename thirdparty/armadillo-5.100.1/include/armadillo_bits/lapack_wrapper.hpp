// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2012 NICTA (www.nicta.com.au)
// Copyright (C) 2009 Edmund Highcock
// Copyright (C) 2011 James Sanders
// Copyright (C) 2012 Eric Jon Sundstrom
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



#ifdef ARMA_USE_LAPACK


//! \namespace lapack namespace for LAPACK functions
namespace lapack
  {
  
  
  template<typename eT>
  inline
  void
  getrf(blas_int* m, blas_int* n, eT* a, blas_int* lda, blas_int* ipiv, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_sgetrf)(m, n, (T*)a, lda, ipiv, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dgetrf)(m, n, (T*)a, lda, ipiv, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_cgetrf)(m, n, (T*)a, lda, ipiv, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_zgetrf)(m, n, (T*)a, lda, ipiv, info);
      }
    }
    
    
    
  template<typename eT>
  inline
  void
  getri(blas_int* n,  eT* a, blas_int* lda, blas_int* ipiv, eT* work, blas_int* lwork, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_sgetri)(n, (T*)a, lda, ipiv, (T*)work, lwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dgetri)(n, (T*)a, lda, ipiv, (T*)work, lwork, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_cgetri)(n, (T*)a, lda, ipiv, (T*)work, lwork, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_zgetri)(n, (T*)a, lda, ipiv, (T*)work, lwork, info);
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  trtri(char* uplo, char* diag, blas_int* n, eT* a, blas_int* lda, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_strtri)(uplo, diag, n, (T*)a, lda, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dtrtri)(uplo, diag, n, (T*)a, lda, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_ctrtri)(uplo, diag, n, (T*)a, lda, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_ztrtri)(uplo, diag, n, (T*)a, lda, info);
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  syev(char* jobz, char* uplo, blas_int* n, eT* a, blas_int* lda, eT* w,  eT* work, blas_int* lwork, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_ssyev)(jobz, uplo, n, (T*)a, lda, (T*)w, (T*)work, lwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dsyev)(jobz, uplo, n, (T*)a, lda, (T*)w, (T*)work, lwork, info);
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  syevd(char* jobz, char* uplo, blas_int* n, eT* a, blas_int* lda, eT* w,  eT* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_ssyevd)(jobz, uplo, n, (T*)a, lda, (T*)w, (T*)work, lwork, iwork, liwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dsyevd)(jobz, uplo, n, (T*)a, lda, (T*)w, (T*)work, lwork, iwork, liwork, info);
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  heev
    (
    char* jobz, char* uplo, blas_int* n,
    eT* a, blas_int* lda, typename eT::value_type* w,
    eT* work, blas_int* lwork, typename eT::value_type* rwork,
    blas_int* info
    )
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef float T;
      typedef typename std::complex<T> cx_T;
      arma_fortran(arma_cheev)(jobz, uplo, n, (cx_T*)a, lda, (T*)w, (cx_T*)work, lwork, (T*)rwork, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef double T;
      typedef typename std::complex<T> cx_T;
      arma_fortran(arma_zheev)(jobz, uplo, n, (cx_T*)a, lda, (T*)w, (cx_T*)work, lwork, (T*)rwork, info);
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  heevd
    (
    char* jobz, char* uplo, blas_int* n,
    eT* a, blas_int* lda, typename eT::value_type* w,
    eT* work, blas_int* lwork, typename eT::value_type* rwork, 
    blas_int* lrwork, blas_int* iwork, blas_int* liwork,
    blas_int* info
    )
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef float T;
      typedef typename std::complex<T> cx_T;
      arma_fortran(arma_cheevd)(jobz, uplo, n, (cx_T*)a, lda, (T*)w, (cx_T*)work, lwork, (T*)rwork, lrwork, iwork, liwork, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef double T;
      typedef typename std::complex<T> cx_T;
      arma_fortran(arma_zheevd)(jobz, uplo, n, (cx_T*)a, lda, (T*)w, (cx_T*)work, lwork, (T*)rwork, lrwork, iwork, liwork, info);
      }
    }
  
	
	
  template<typename eT>
  inline
  void
  geev
    (
    char* jobvl, char* jobvr, blas_int* n, 
    eT* a, blas_int* lda, eT* wr, eT* wi, eT* vl, 
    blas_int* ldvl, eT* vr, blas_int* ldvr, 
    eT* work, blas_int* lwork,
    blas_int* info
    )
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));

    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_sgeev)(jobvl, jobvr, n,  (T*)a, lda, (T*)wr, (T*)wi, (T*)vl, ldvl, (T*)vr, ldvr, (T*)work, lwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dgeev)(jobvl, jobvr, n,  (T*)a, lda, (T*)wr, (T*)wi, (T*)vl, ldvl, (T*)vr, ldvr, (T*)work, lwork, info);
      }
    }


  template<typename eT>
  inline
  void
  cx_geev
    (
    char* jobvl, char* jobvr, blas_int* n, 
    eT* a, blas_int* lda, eT* w, 
    eT* vl, blas_int* ldvl, 
    eT* vr, blas_int* ldvr, 
    eT* work, blas_int* lwork, typename eT::value_type* rwork, 
    blas_int* info
    )
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef float T;
      typedef typename std::complex<T> cx_T;
      arma_fortran(arma_cgeev)(jobvl, jobvr, n, (cx_T*)a, lda, (cx_T*)w, (cx_T*)vl, ldvl, (cx_T*)vr, ldvr, (cx_T*)work, lwork, (T*)rwork, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef double T;
      typedef typename std::complex<T> cx_T;
      arma_fortran(arma_zgeev)(jobvl, jobvr, n, (cx_T*)a, lda, (cx_T*)w, (cx_T*)vl, ldvl, (cx_T*)vr, ldvr, (cx_T*)work, lwork, (T*)rwork, info);
      }
    }
  
  
  template<typename eT>
  inline
  void
  ggev
    (
    char* jobvl, char* jobvr, blas_int* n,
    eT* a, blas_int* lda, eT* b, blas_int* ldb,
    eT* alphar, eT* alphai, eT* beta,
    eT* vl, blas_int* ldvl, eT* vr, blas_int* ldvr,
    eT* work, blas_int* lwork,
    blas_int* info
    )
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_sggev)(jobvl, jobvr, n, (T*)a, lda, (T*)b, ldb, (T*)alphar, (T*)alphai, (T*)beta, (T*)vl, ldvl, (T*)vr, ldvr, (T*)work, lwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dggev)(jobvl, jobvr, n, (T*)a, lda, (T*)b, ldb, (T*)alphar, (T*)alphai, (T*)beta, (T*)vl, ldvl, (T*)vr, ldvr, (T*)work, lwork, info);
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  cx_ggev
    (
    char* jobvl, char* jobvr, blas_int* n,
    eT* a, blas_int* lda, eT* b, blas_int* ldb,
    eT* alpha, eT* beta,
    eT* vl, blas_int* ldvl, eT* vr, blas_int* ldvr,
    eT* work, blas_int* lwork, typename eT::value_type* rwork,
    blas_int* info
    )
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef float T;
      typedef typename std::complex<T> cx_T;
      arma_fortran(arma_cggev)(jobvl, jobvr, n, (cx_T*)a, lda, (cx_T*)b, ldb, (cx_T*)alpha, (cx_T*)beta, (cx_T*)vl, ldvl, (cx_T*)vr, ldvr, (cx_T*)work, lwork, (T*)rwork, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef double T;
      typedef typename std::complex<T> cx_T;
      arma_fortran(arma_zggev)(jobvl, jobvr, n, (cx_T*)a, lda, (cx_T*)b, ldb, (cx_T*)alpha, (cx_T*)beta, (cx_T*)vl, ldvl, (cx_T*)vr, ldvr, (cx_T*)work, lwork, (T*)rwork, info);
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  potrf(char* uplo, blas_int* n, eT* a, blas_int* lda, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_spotrf)(uplo, n, (T*)a, lda, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dpotrf)(uplo, n, (T*)a, lda, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_cpotrf)(uplo, n, (T*)a, lda, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_zpotrf)(uplo, n, (T*)a, lda, info);
      }
    
    }
  
  
  
  template<typename eT>
  inline
  void
  potri(char* uplo, blas_int* n, eT* a, blas_int* lda, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_spotri)(uplo, n, (T*)a, lda, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dpotri)(uplo, n, (T*)a, lda, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_cpotri)(uplo, n, (T*)a, lda, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_zpotri)(uplo, n, (T*)a, lda, info);
      }
    
    }
  
  
  
  template<typename eT>
  inline
  void
  geqrf(blas_int* m, blas_int* n, eT* a, blas_int* lda, eT* tau, eT* work, blas_int* lwork, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_sgeqrf)(m, n, (T*)a, lda, (T*)tau, (T*)work, lwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dgeqrf)(m, n, (T*)a, lda, (T*)tau, (T*)work, lwork, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_cgeqrf)(m, n, (T*)a, lda, (T*)tau, (T*)work, lwork, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_zgeqrf)(m, n, (T*)a, lda, (T*)tau, (T*)work, lwork, info);
      }
    
    }
  
  
  
  template<typename eT>
  inline
  void
  orgqr(blas_int* m, blas_int* n, blas_int* k, eT* a, blas_int* lda, eT* tau, eT* work, blas_int* lwork, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_sorgqr)(m, n, k, (T*)a, lda, (T*)tau, (T*)work, lwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dorgqr)(m, n, k, (T*)a, lda, (T*)tau, (T*)work, lwork, info);
      }
    }


  
  template<typename eT>
  inline
  void
  ungqr(blas_int* m, blas_int* n, blas_int* k, eT* a, blas_int* lda, eT* tau, eT* work, blas_int* lwork, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_cungqr)(m, n, k, (T*)a, lda, (T*)tau, (T*)work, lwork, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_zungqr)(m, n, k, (T*)a, lda, (T*)tau, (T*)work, lwork, info);
      }
    }
  
  
  template<typename eT>
  inline
  void
  gesvd
    (
    char* jobu, char* jobvt, blas_int* m, blas_int* n, eT* a, blas_int* lda,
    eT* s, eT* u, blas_int* ldu, eT* vt, blas_int* ldvt,
    eT* work, blas_int* lwork, blas_int* info
    )
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_sgesvd)(jobu, jobvt, m, n, (T*)a, lda, (T*)s, (T*)u, ldu, (T*)vt, ldvt, (T*)work, lwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dgesvd)(jobu, jobvt, m, n, (T*)a, lda, (T*)s, (T*)u, ldu, (T*)vt, ldvt, (T*)work, lwork, info);
      }
    }
  
  
  
  template<typename T>
  inline
  void
  cx_gesvd
    (
    char* jobu, char* jobvt, blas_int* m, blas_int* n, std::complex<T>* a, blas_int* lda,
    T* s, std::complex<T>* u, blas_int* ldu, std::complex<T>* vt, blas_int* ldvt, 
    std::complex<T>* work, blas_int* lwork, T* rwork, blas_int* info
    )
    {
    arma_type_check(( is_supported_blas_type<T>::value == false ));
    arma_type_check(( is_supported_blas_type< std::complex<T> >::value == false ));
    
    if(is_float<T>::value == true)
      {
      typedef float bT;
      arma_fortran(arma_cgesvd)
        (
        jobu, jobvt, m, n, (std::complex<bT>*)a, lda,
        (bT*)s, (std::complex<bT>*)u, ldu, (std::complex<bT>*)vt, ldvt,
        (std::complex<bT>*)work, lwork, (bT*)rwork, info
        );
      }
    else
    if(is_double<T>::value == true)
      {
      typedef double bT;
      arma_fortran(arma_zgesvd)
        (
        jobu, jobvt, m, n, (std::complex<bT>*)a, lda,
        (bT*)s, (std::complex<bT>*)u, ldu, (std::complex<bT>*)vt, ldvt,
        (std::complex<bT>*)work, lwork, (bT*)rwork, info
        );
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  gesdd
    (
    char* jobz, blas_int* m, blas_int* n,
    eT* a, blas_int* lda, eT* s, eT* u, blas_int* ldu, eT* vt, blas_int* ldvt,
    eT* work, blas_int* lwork, blas_int* iwork, blas_int* info
    )
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_sgesdd)(jobz, m, n, (T*)a, lda, (T*)s, (T*)u, ldu, (T*)vt, ldvt, (T*)work, lwork, iwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dgesdd)(jobz, m, n, (T*)a, lda, (T*)s, (T*)u, ldu, (T*)vt, ldvt, (T*)work, lwork, iwork, info);
      }
    }
  
  
  
  template<typename T>
  inline
  void
  cx_gesdd
    (
    char* jobz, blas_int* m, blas_int* n,
    std::complex<T>* a, blas_int* lda, T* s, std::complex<T>* u, blas_int* ldu, std::complex<T>* vt, blas_int* ldvt,
    std::complex<T>* work, blas_int* lwork, T* rwork, blas_int* iwork, blas_int* info
    )
    {
    arma_type_check(( is_supported_blas_type<T>::value == false ));
    arma_type_check(( is_supported_blas_type< std::complex<T> >::value == false ));
    
    if(is_float<T>::value == true)
      {
      typedef float bT;
      arma_fortran(arma_cgesdd)
        (
        jobz, m, n,
        (std::complex<bT>*)a, lda, (bT*)s, (std::complex<bT>*)u, ldu, (std::complex<bT>*)vt, ldvt,
        (std::complex<bT>*)work, lwork, (bT*)rwork, iwork, info
        );
      }
    else
    if(is_double<T>::value == true)
      {
      typedef double bT;
      arma_fortran(arma_zgesdd)
        (
        jobz, m, n,
        (std::complex<bT>*)a, lda, (bT*)s, (std::complex<bT>*)u, ldu, (std::complex<bT>*)vt, ldvt,
        (std::complex<bT>*)work, lwork, (bT*)rwork, iwork, info
        );
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  gesv(blas_int* n, blas_int* nrhs, eT* a, blas_int* lda, blas_int* ipiv, eT* b, blas_int* ldb, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_sgesv)(n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dgesv)(n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_cgesv)(n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_zgesv)(n, nrhs, (T*)a, lda, ipiv, (T*)b, ldb, info);
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  gels(char* trans, blas_int* m, blas_int* n, blas_int* nrhs, eT* a, blas_int* lda, eT* b, blas_int* ldb, eT* work, blas_int* lwork, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_sgels)(trans, m, n, nrhs, (T*)a, lda, (T*)b, ldb, (T*)work, lwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dgels)(trans, m, n, nrhs, (T*)a, lda, (T*)b, ldb, (T*)work, lwork, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_cgels)(trans, m, n, nrhs, (T*)a, lda, (T*)b, ldb, (T*)work, lwork, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_zgels)(trans, m, n, nrhs, (T*)a, lda, (T*)b, ldb, (T*)work, lwork, info);
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  trtrs(char* uplo, char* trans, char* diag, blas_int* n, blas_int* nrhs, const eT* a, blas_int* lda, eT* b, blas_int* ldb, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_strtrs)(uplo, trans, diag, n, nrhs, (T*)a, lda, (T*)b, ldb, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dtrtrs)(uplo, trans, diag, n, nrhs, (T*)a, lda, (T*)b, ldb, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_ctrtrs)(uplo, trans, diag, n, nrhs, (T*)a, lda, (T*)b, ldb, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_ztrtrs)(uplo, trans, diag, n, nrhs, (T*)a, lda, (T*)b, ldb, info);
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  gees(char* jobvs, char* sort, blas_int* select, blas_int* n, eT* a, blas_int* lda, blas_int* sdim, eT* wr, eT* wi, eT* vs, blas_int* ldvs, eT* work, blas_int* lwork, blas_int* bwork, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_sgees)(jobvs, sort, select, n, (T*)a, lda, sdim, (T*)wr, (T*)wi, (T*)vs, ldvs, (T*)work, lwork, bwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dgees)(jobvs, sort, select, n, (T*)a, lda, sdim, (T*)wr, (T*)wi, (T*)vs, ldvs, (T*)work, lwork, bwork, info);
      }
    }
  
  
  
  template<typename T>
  inline
  void
  cx_gees(char* jobvs, char* sort, blas_int* select, blas_int* n, std::complex<T>* a, blas_int* lda, blas_int* sdim, std::complex<T>* w, std::complex<T>* vs, blas_int* ldvs, std::complex<T>* work, blas_int* lwork, T* rwork, blas_int* bwork, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<T>::value == false ));
    arma_type_check(( is_supported_blas_type< std::complex<T> >::value == false ));
    
    if(is_float<T>::value == true)
      {
      typedef float bT;
      typedef std::complex<bT> cT;
      arma_fortran(arma_cgees)(jobvs, sort, select, n, (cT*)a, lda, sdim, (cT*)w, (cT*)vs, ldvs, (cT*)work, lwork, (bT*)rwork, bwork, info);
      }
    else
    if(is_double<T>::value == true)
      {
      typedef double bT;
      typedef std::complex<bT> cT;
      arma_fortran(arma_zgees)(jobvs, sort, select, n, (cT*)a, lda, sdim, (cT*)w, (cT*)vs, ldvs, (cT*)work, lwork, (bT*)rwork, bwork, info);
      }
    }
  
  
  
  template<typename eT>
  inline
  void
  trsyl(char* transa, char* transb, blas_int* isgn, blas_int* m, blas_int* n, const eT* a, blas_int* lda, const eT* b, blas_int* ldb, eT* c, blas_int* ldc, eT* scale, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_strsyl)(transa, transb, isgn, m, n, (T*)a, lda, (T*)b, ldb, (T*)c, ldc, (T*)scale, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dtrsyl)(transa, transb, isgn, m, n, (T*)a, lda, (T*)b, ldb, (T*)c, ldc, (T*)scale, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_ctrsyl)(transa, transb, isgn, m, n, (T*)a, lda, (T*)b, ldb, (T*)c, ldc, (float*)scale, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_ztrsyl)(transa, transb, isgn, m, n, (T*)a, lda, (T*)b, ldb, (T*)c, ldc, (double*)scale, info);
      }
    }
  
  
  template<typename eT>
  inline
  void
  sytrf(char* uplo, blas_int* n, eT* a, blas_int* lda, blas_int* ipiv, eT* work, blas_int* lwork, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_ssytrf)(uplo, n, (T*)a, lda, ipiv, (T*)work, lwork, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dsytrf)(uplo, n, (T*)a, lda, ipiv, (T*)work, lwork, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_csytrf)(uplo, n, (T*)a, lda, ipiv, (T*)work, lwork, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_zsytrf)(uplo, n, (T*)a, lda, ipiv, (T*)work, lwork, info);
      }
    }
  
  
  template<typename eT>
  inline
  void
  sytri(char* uplo, blas_int* n, eT* a, blas_int* lda, blas_int* ipiv, eT* work, blas_int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value == true)
      {
      typedef float T;
      arma_fortran(arma_ssytri)(uplo, n, (T*)a, lda, ipiv, (T*)work, info);
      }
    else
    if(is_double<eT>::value == true)
      {
      typedef double T;
      arma_fortran(arma_dsytri)(uplo, n, (T*)a, lda, ipiv, (T*)work, info);
      }
    else
    if(is_supported_complex_float<eT>::value == true)
      {
      typedef std::complex<float> T;
      arma_fortran(arma_csytri)(uplo, n, (T*)a, lda, ipiv, (T*)work, info);
      }
    else
    if(is_supported_complex_double<eT>::value == true)
      {
      typedef std::complex<double> T;
      arma_fortran(arma_zsytri)(uplo, n, (T*)a, lda, ipiv, (T*)work, info);
      }
    }
  
  
  }


#endif
