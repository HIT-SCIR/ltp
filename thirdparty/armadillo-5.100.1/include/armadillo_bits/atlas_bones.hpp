// Copyright (C) 2008-2013 Conrad Sanderson
// Copyright (C) 2008-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifdef ARMA_USE_ATLAS


//! \namespace atlas namespace for ATLAS functions (imported from the global namespace)
namespace atlas
  {
  
  using ::CblasColMajor;
  using ::CblasNoTrans;
  using ::CblasTrans;
  using ::CblasConjTrans;
  using ::CblasLower;
  using ::CblasUpper;
  
  #if defined(ARMA_USE_WRAPPER)
  extern "C"
    {
    
    float  wrapper_cblas_sdot(const int N, const float  *X, const int incX, const float  *Y, const int incY);
    double wrapper_cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);
    
    void wrapper_cblas_cdotu_sub(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotu);
    void wrapper_cblas_zdotu_sub(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotu);
    
    
    void wrapper_cblas_sgemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha,
                             const float *A, const int lda, const float *X, const int incX, const float beta, float *Y, const int incY);
    
    void wrapper_cblas_dgemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const double alpha,
                             const double *A, const int lda, const double *X, const int incX, const double beta, double *Y, const int incY);
    
    void wrapper_cblas_cgemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const void *alpha,
                             const void *A, const int lda, const void *X, const int incX, const void *beta, void *Y, const int incY);
    
    void wrapper_cblas_zgemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const void *alpha,
                             const void *A, const int lda, const void *X, const int incX, const void *beta, void *Y, const int incY);
    
    
    
    void wrapper_cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                             const int M, const int N, const int K, const float alpha,
                             const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
    
    void wrapper_cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                             const int M, const int N, const int K, const double alpha,
                             const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
    
    void wrapper_cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                             const int M, const int N, const int K, const void *alpha,
                             const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);
    
    void wrapper_cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                             const int M, const int N, const int K, const void *alpha,
                             const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);
    
    
    
    void wrapper_cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                             const int N, const int K, const float alpha,
                             const float *A, const int lda, const float beta, float *C, const int ldc);
    
    void wrapper_cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                             const int N, const int K, const double alpha,
                             const double *A, const int lda, const double beta, double *C, const int ldc);
    
    
    
    void wrapper_cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                             const int N, const int K, const float alpha,
                             const void *A, const int lda, const float beta, void *C, const int ldc);
    
    void wrapper_cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                             const int N, const int K, const double alpha,
                             const void *A, const int lda, const double beta, void *C, const int ldc);
    
    
    
    int wrapper_clapack_sgetrf(const enum CBLAS_ORDER Order, const int M, const int N, float  *A, const int lda, int *ipiv);
    int wrapper_clapack_dgetrf(const enum CBLAS_ORDER Order, const int M, const int N, double *A, const int lda, int *ipiv);
    int wrapper_clapack_cgetrf(const enum CBLAS_ORDER Order, const int M, const int N, void   *A, const int lda, int *ipiv);
    int wrapper_clapack_zgetrf(const enum CBLAS_ORDER Order, const int M, const int N, void   *A, const int lda, int *ipiv);
    
    int wrapper_clapack_sgetri(const enum CBLAS_ORDER Order, const int N, float  *A, const int lda, const int *ipiv);
    int wrapper_clapack_dgetri(const enum CBLAS_ORDER Order, const int N, double *A, const int lda, const int *ipiv);
    int wrapper_clapack_cgetri(const enum CBLAS_ORDER Order, const int N, void   *A, const int lda, const int *ipiv);
    int wrapper_clapack_zgetri(const enum CBLAS_ORDER Order, const int N, void   *A, const int lda, const int *ipiv);

    int wrapper_clapack_sgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS, float  *A, const int lda, int *ipiv, float  *B, const int ldb);
    int wrapper_clapack_dgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS, double *A, const int lda, int *ipiv, double *B, const int ldb);
    int wrapper_clapack_cgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS, void   *A, const int lda, int *ipiv, void   *B, const int ldb);
    int wrapper_clapack_zgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS, void   *A, const int lda, int *ipiv, void   *B, const int ldb);
    
    }
  #endif
  
  }


#endif
