// Copyright (C) 2013 Ryan Curtin
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifdef ARMA_USE_ARPACK

// I'm not sure this is necessary.
#if !defined(ARMA_BLAS_CAPITALS)

  #define arma_snaupd snaupd
  #define arma_dnaupd dnaupd
  #define arma_cnaupd cnaupd
  #define arma_znaupd znaupd

  #define arma_sneupd sneupd
  #define arma_dneupd dneupd
  #define arma_cneupd cneupd
  #define arma_zneupd zneupd

  #define arma_ssaupd ssaupd
  #define arma_dsaupd dsaupd

  #define arma_sseupd sseupd
  #define arma_dseupd dseupd

#else

  #define arma_snaupd SNAUPD
  #define arma_dnaupd DNAUPD
  #define arma_cnaupd CNAUPD
  #define arma_znaupd ZNAUPD

  #define arma_sneupd SNEUPD
  #define arma_dneupd DNEUPD
  #define arma_cneupd CNEUPD
  #define arma_zneupd ZNEUPD

  #define arma_ssaupd SSAUPD
  #define arma_dsaupd DSAUPD

  #define arma_sseupd SSEUPD
  #define arma_dseupd DSEUPD

#endif

extern "C"
  {
  // eigendecomposition of non-symmetric positive semi-definite matrices
  void arma_fortran(arma_snaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev,  float* tol,  float* resid, blas_int* ncv,  float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr,  float* workd,  float* workl, blas_int* lworkl, blas_int* info);
  void arma_fortran(arma_dnaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info);
  void arma_fortran(arma_cnaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev,  float* tol,   void* resid, blas_int* ncv,   void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr,   void* workd,   void* workl, blas_int* lworkl, float* rwork, blas_int* info);
  void arma_fortran(arma_znaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol,   void* resid, blas_int* ncv,   void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr,   void* workd,   void* workl, blas_int* lworkl, double* rwork, blas_int* info);

  // eigendecomposition of symmetric positive semi-definite matrices
  void arma_fortran(arma_ssaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev,  float* tol,  float* resid, blas_int* ncv,  float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr,  float* workd,  float* workl, blas_int* lworkl, blas_int* info);
  void arma_fortran(arma_dsaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info);

  // recovery of eigenvectors after naupd(); uses blas_int for LOGICAL types
  void arma_fortran(arma_sneupd)(blas_int* rvec, char* howmny, blas_int* select,  float* dr,  float* di,  float* z, blas_int* ldz,  float* sigmar,  float* sigmai,  float* workev, char* bmat, blas_int* n, char* which, blas_int* nev,  float* tol,  float* resid, blas_int* ncv,  float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr,  float* workd,  float* workl, blas_int* lworkl, blas_int* info);
  void arma_fortran(arma_dneupd)(blas_int* rvec, char* howmny, blas_int* select, double* dr, double* di, double* z, blas_int* ldz, double* sigmar, double* sigmai, double* workev, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info);
  void arma_fortran(arma_cneupd)(blas_int* rvec, char* howmny, blas_int* select, void* d,   void* z, blas_int* ldz,   void* sigma,   void* workev, char* bmat, blas_int* n, char* which, blas_int* nev,  float* tol,   void* resid, blas_int* ncv,   void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr,   void* workd, void* workl, blas_int* lworkl,  float* rwork, blas_int* info);
  void arma_fortran(arma_zneupd)(blas_int* rvec, char* howmny, blas_int* select, void* d,   void* z, blas_int* ldz,   void* sigma,   void* workev, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol,   void* resid, blas_int* ncv,   void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr,   void* workd, void* workl, blas_int* lworkl, double* rwork, blas_int* info);

  // recovery of eigenvectors after saupd(); uses blas_int for LOGICAL types
  void arma_fortran(arma_sseupd)(blas_int* rvec, char* howmny, blas_int* select,  float* d,  float* z, blas_int* ldz,  float* sigma, char* bmat, blas_int* n, char* which, blas_int* nev,  float* tol,  float* resid, blas_int* ncv,  float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr,  float* workd,  float* workl, blas_int* lworkl, blas_int* info);
  void arma_fortran(arma_dseupd)(blas_int* rvec, char* howmny, blas_int* select, double* d, double* z, blas_int* ldz, double* sigma, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info);
  }

#endif
