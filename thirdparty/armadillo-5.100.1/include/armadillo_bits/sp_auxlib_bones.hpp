// Copyright (C) 2013-2015 Ryan Curtin
// Copyright (C) 2013-2015 Conrad Sanderson
// Copyright (C) 2013-2015 NICTA
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup sp_auxlib
//! @{


//! wrapper for accesing external functions for sparse matrices defined in ARPACK
class sp_auxlib
  {
  public:
  
  enum form_type
    {
    form_none, form_lm, form_sm, form_lr, form_la, form_sr, form_li, form_si, form_sa
    };
  
  inline static form_type interpret_form_str(const char* form_str);
  
  //
  // eigs_sym() via ARPACK
  
  template<typename eT, typename T1>
  inline static bool eigs_sym(Col<eT>& eigval, Mat<eT>& eigvec, const SpBase<eT, T1>& X, const uword n_eigvals, const char* form_str, const eT default_tol);
  
  
  //
  // eigs_gen() via ARPACK
  
  template<typename T, typename T1>
  inline static bool eigs_gen(Col< std::complex<T> >& eigval, Mat< std::complex<T> >& eigvec, const SpBase<T, T1>& X, const uword n_eigvals, const char* form_str, const T default_tol);
  
  template<typename T, typename T1>
  inline static bool eigs_gen(Col< std::complex<T> >& eigval, Mat< std::complex<T> >& eigvec, const SpBase< std::complex<T>, T1>& X, const uword n_eigvals, const char* form_str, const T default_tol);
  
  
  //
  // spsolve() via SuperLU
  
  template<typename T1, typename T2>
  inline static bool spsolve(Mat<typename T1::elem_type>& out, const SpBase<typename T1::elem_type, T1>& A, const Base<typename T1::elem_type, T2>& B, const superlu_opts& user_opts);
  
  #if defined(ARMA_USE_SUPERLU)
    template<typename eT>
    inline static bool convert_to_supermatrix(superlu::SuperMatrix& out, const SpMat<eT>& A);
    
    template<typename eT>
    inline static bool convert_to_supermatrix(superlu::SuperMatrix& out, const Mat<eT>& A);
    
    inline static void destroy_supermatrix(superlu::SuperMatrix& out);
  #endif
  
  
  
  private:
  
  // calls arpack saupd()/naupd() because the code is so similar for each
  // all of the extra variables are later used by seupd()/neupd(), but those
  // functions are very different and we can't combine their code
  
  template<typename eT, typename T, typename T1>
  inline static void run_aupd
    (
    const uword n_eigvals, char* which, const SpProxy<T1>& p, const bool sym,
    blas_int& n, eT& tol,
    podarray<T>& resid, blas_int& ncv, podarray<T>& v, blas_int& ldv,
    podarray<blas_int>& iparam, podarray<blas_int>& ipntr,
    podarray<T>& workd, podarray<T>& workl, blas_int& lworkl, podarray<eT>& rwork,
    blas_int& info
    );

  };
