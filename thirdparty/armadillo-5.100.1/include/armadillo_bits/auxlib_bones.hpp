// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// Copyright (C) 2009 Edmund Highcock
// Copyright (C) 2011 James Sanders
// Copyright (C) 2012 Eric Jon Sundstrom
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup auxlib
//! @{


//! wrapper for accessing external functions defined in ATLAS, LAPACK or BLAS libraries
class auxlib
  {
  public:
  
  
  template<const uword row, const uword col>
  struct pos
    {
    static const uword n2 = row + col*2;
    static const uword n3 = row + col*3;
    static const uword n4 = row + col*4;
    };
  
  
  //
  // inv
  
  template<typename eT, typename T1>
  inline static bool inv(Mat<eT>& out, const Base<eT,T1>& X, const bool slow = false);
  
  template<typename eT>
  inline static bool inv(Mat<eT>& out, const Mat<eT>& A, const bool slow = false);
  
  template<typename eT>
  inline static bool inv_noalias_tinymat(Mat<eT>& out, const Mat<eT>& X, const uword N);
  
  template<typename eT>
  inline static bool inv_inplace_lapack(Mat<eT>& out);
  
  
  //
  // inv_tr
  
  template<typename eT, typename T1>
  inline static bool inv_tr(Mat<eT>& out, const Base<eT,T1>& X, const uword layout);
  
  
  //
  // inv_sym
  
  template<typename eT, typename T1>
  inline static bool inv_sym(Mat<eT>& out, const Base<eT,T1>& X, const uword layout);
  
  
  //
  // inv_sympd
  
  template<typename eT, typename T1>
  inline static bool inv_sympd(Mat<eT>& out, const Base<eT,T1>& X, const uword layout);
  
  
  //
  // det
  
  template<typename eT, typename T1>
  inline static eT det(const Base<eT,T1>& X, const bool slow = false);
  
  template<typename eT>
  inline static eT det_tinymat(const Mat<eT>& X, const uword N);
  
  template<typename eT>
  inline static eT det_lapack(const Mat<eT>& X, const bool make_copy);
  
  
  //
  // log_det
  
  template<typename eT, typename T1>
  inline static bool log_det(eT& out_val, typename get_pod_type<eT>::result& out_sign, const Base<eT,T1>& X);
  
  
  //
  // lu
  
  template<typename eT, typename T1>
  inline static bool lu(Mat<eT>& L, Mat<eT>& U, podarray<blas_int>& ipiv, const Base<eT,T1>& X);
  
  template<typename eT, typename T1>
  inline static bool lu(Mat<eT>& L, Mat<eT>& U, Mat<eT>& P, const Base<eT,T1>& X);
  
  template<typename eT, typename T1>
  inline static bool lu(Mat<eT>& L, Mat<eT>& U, const Base<eT,T1>& X);
  
  
  //
  // eig_sym
  
  template<typename eT, typename T1> 
  inline static bool eig_sym(Col<eT>& eigval, const Base<eT,T1>& X);
  
  template<typename T, typename T1> 
  inline static bool eig_sym(Col<T>& eigval, const Base<std::complex<T>,T1>& X);
  
  template<typename eT, typename T1>
  inline static bool eig_sym(Col<eT>& eigval, Mat<eT>& eigvec, const Base<eT,T1>& X);
  
  template<typename T, typename T1>
  inline static bool eig_sym(Col<T>& eigval, Mat< std::complex<T> >& eigvec, const Base<std::complex<T>,T1>& X);
  
  template<typename eT, typename T1>
  inline static bool eig_sym_dc(Col<eT>& eigval, Mat<eT>& eigvec, const Base<eT,T1>& X);
  
  template<typename T, typename T1>
  inline static bool eig_sym_dc(Col<T>& eigval, Mat< std::complex<T> >& eigvec, const Base<std::complex<T>,T1>& X);
  
  
  //
  // eig_gen
  
  template<typename T, typename T1>
  inline static bool eig_gen(Col< std::complex<T> >& eigval, Mat<T>& l_eigvec, Mat<T>& r_eigvec, const Base<T,T1>& X, const char side);
  
  template<typename T, typename T1>
  inline static bool eig_gen(Col< std::complex<T> >& eigval, Mat< std::complex<T> >& l_eigvec, Mat< std::complex<T> >& r_eigvec, const Base< std::complex<T>, T1 >& X, const char side);
  
  
  //
  // eig_pair
  
  template<typename T, typename T1, typename T2>
  inline static bool eig_pair(Col< std::complex<T> >& eigval, Mat<T>& l_eigvec, Mat<T>& r_eigvec, const Base<T,T1>& X, const Base<T,T2>& Y, const char side);
  
  template<typename T, typename T1, typename T2>
  inline static bool eig_pair(Col< std::complex<T> >& eigval, Mat< std::complex<T> >& l_eigvec, Mat< std::complex<T> >& r_eigvec, const Base< std::complex<T>, T1 >& X, const Base< std::complex<T>, T2 >& Y, const char side);
  
  
  //
  // chol
  
  template<typename eT, typename T1>
  inline static bool chol(Mat<eT>& out, const Base<eT,T1>& X, const uword layout);
  
  
  //
  // qr
  
  template<typename eT, typename T1>
  inline static bool qr(Mat<eT>& Q, Mat<eT>& R, const Base<eT,T1>& X);
  
  template<typename eT, typename T1>
  inline static bool qr_econ(Mat<eT>& Q, Mat<eT>& R, const Base<eT,T1>& X);
  
  
  //
  // svd
  
  template<typename eT, typename T1>
  inline static bool svd(Col<eT>& S, const Base<eT,T1>& X, uword& n_rows, uword& n_cols);
  
  template<typename T, typename T1>
  inline static bool svd(Col<T>& S, const Base<std::complex<T>, T1>& X, uword& n_rows, uword& n_cols);
  
  template<typename eT, typename T1>
  inline static bool svd(Col<eT>& S, const Base<eT,T1>& X);
  
  template<typename T, typename T1>
  inline static bool svd(Col<T>& S, const Base<std::complex<T>, T1>& X);
  
  template<typename eT, typename T1>
  inline static bool svd(Mat<eT>& U, Col<eT>& S, Mat<eT>& V, const Base<eT,T1>& X);
  
  template<typename T, typename T1>
  inline static bool svd(Mat< std::complex<T> >& U, Col<T>& S, Mat< std::complex<T> >& V, const Base< std::complex<T>, T1>& X);
  
  template<typename eT, typename T1>
  inline static bool svd_econ(Mat<eT>& U, Col<eT>& S, Mat<eT>& V, const Base<eT,T1>& X, const char mode);
  
  template<typename T, typename T1>
  inline static bool svd_econ(Mat< std::complex<T> >& U, Col<T>& S, Mat< std::complex<T> >& V, const Base< std::complex<T>, T1>& X, const char mode);
  
  
  // EXPERIMENTAL
  template<typename eT, typename T1>
  inline static bool svd_dc(Col<eT>& S, const Base<eT,T1>& X, uword& n_rows, uword& n_cols);
  
  // EXPERIMENTAL
  template<typename T, typename T1>
  inline static bool svd_dc(Col<T>& S, const Base<std::complex<T>, T1>& X, uword& n_rows, uword& n_cols);
  
  // EXPERIMENTAL
  template<typename eT, typename T1>
  inline static bool svd_dc(Col<eT>& S, const Base<eT,T1>& X);
  
  // EXPERIMENTAL
  template<typename T, typename T1>
  inline static bool svd_dc(Col<T>& S, const Base<std::complex<T>, T1>& X);
  
  
  template<typename eT, typename T1>
  inline static bool svd_dc(Mat<eT>& U, Col<eT>& S, Mat<eT>& V, const Base<eT,T1>& X);
  
  template<typename T, typename T1>
  inline static bool svd_dc(Mat< std::complex<T> >& U, Col<T>& S, Mat< std::complex<T> >& V, const Base< std::complex<T>, T1>& X);
  
  template<typename eT, typename T1>
  inline static bool svd_dc_econ(Mat<eT>& U, Col<eT>& S, Mat<eT>& V, const Base<eT,T1>& X);
  
  template<typename T, typename T1>
  inline static bool svd_dc_econ(Mat< std::complex<T> >& U, Col<T>& S, Mat< std::complex<T> >& V, const Base< std::complex<T>, T1>& X);
  
  
  //
  // solve
  
  template<typename eT, typename T1>
  inline static bool solve   (Mat<eT>& out, Mat<eT>& A, const Base<eT,T1>& X, const bool slow = false);
  
  template<typename eT, typename T1>
  inline static bool solve_od(Mat<eT>& out, Mat<eT>& A, const Base<eT,T1>& X);
  
  template<typename eT, typename T1>
  inline static bool solve_ud(Mat<eT>& out, Mat<eT>& A, const Base<eT,T1>& X);
  
  
  //
  // solve_tr
  
  template<typename eT>
  inline static bool solve_tr(Mat<eT>& out, const Mat<eT>& A, const Mat<eT>& B, const uword layout);


  //
  // Schur decomposition
  
  template<typename eT>
  inline static bool schur_dec(Mat<eT>& Z, Mat<eT>& T, const Mat<eT>& A);
  
  template<typename cT>
  inline static bool schur_dec(Mat<std::complex<cT> >& Z, Mat<std::complex<cT> >& T, const Mat<std::complex<cT> >& A);
  
  
  //
  // syl (solution of the Sylvester equation AX + XB = C)
  
  template<typename eT>
  inline static bool syl(Mat<eT>& X, const Mat<eT>& A, const Mat<eT>& B, const Mat<eT>& C);
  
  
  //
  // lyap (solution of the continuous Lyapunov equation AX + XA^H + Q = 0)
  
  template<typename eT>
  inline static bool lyap(Mat<eT>& X, const Mat<eT>& A, const Mat<eT>& Q);
  
  
  //
  // dlyap (solution of the discrete Lyapunov equation AXA^H - X + Q = 0)
  
  template<typename eT>
  inline static bool dlyap(Mat<eT>& X, const Mat<eT>& A, const Mat<eT>& Q);
  };


//! @}
