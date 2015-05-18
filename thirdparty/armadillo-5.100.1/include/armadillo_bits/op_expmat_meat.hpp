// Copyright (C) 2014 Conrad Sanderson
// Copyright (C) 2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup op_expmat
//! @{


//! implementation based on:
//! Cleve Moler, Charles Van Loan.
//! Nineteen Dubious Ways to Compute the Exponential of a Matrix, Twenty-Five Years Later.
//! SIAM Review, Vol. 45, No. 1, 2003, pp. 3-49.
//! http://dx.doi.org/10.1137/S00361445024180


template<typename T1>
inline
void
op_expmat::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_expmat>& expr)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  
  if(is_op_diagmat<T1>::value)
    {
    out = expr.m;  // force the evaluation of diagmat()
    
    const uword n_rows = out.n_rows;
    
    for(uword i=0; i<n_rows; ++i)
      {
      out.at(i,i) = std::exp( out.at(i,i) );
      }
    }
  else
    {
    const unwrap<T1>   tmp(expr.m);
    const Mat<eT>& A = tmp.M;
    
    arma_debug_check( (A.is_square() == false), "expmat(): given matrix is not square sized" );
    
    const T norm_val = arma::norm(A, "inf");
    
    const double log2_val = (norm_val > T(0)) ? double(eop_aux::log2(norm_val)) : double(0);
    
    int exponent = int(0);  std::frexp(log2_val, &exponent);
    
    const uword s = uword( (std::max)(int(0), exponent + int(1)) );
    
    const Mat<eT> AA = A / eT(eop_aux::pow(double(2), double(s)));
    
    T c = T(0.5);
    
    Mat<eT> E(AA.n_rows, AA.n_rows, fill::eye);  E += c * AA;
    Mat<eT> D(AA.n_rows, AA.n_rows, fill::eye);  D -= c * AA;
    
    Mat<eT> X = AA;
    
    bool positive = true;
    
    const uword N = 6;
    
    for(uword i = 2; i <= N; ++i)
      {
      c = c * T(N - i + 1) / T(i * (2*N - i + 1));
      
      X = AA * X;
      
      E += c * X;
      
      if(positive)  { D += c * X; }  else  { D -= c * X; }
      
      positive = (positive) ? false : true;
      }
    
    out = solve(D, E);
    
    for(uword i=0; i < s; ++i)  { out = out * out; }
    }
  }



//! @}
