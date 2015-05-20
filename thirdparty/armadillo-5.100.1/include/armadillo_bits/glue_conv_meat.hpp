// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup glue_conv
//! @{


//! rudimentary implementation of the convolution operation

template<typename T1, typename T2>
inline
void
glue_conv::apply(Mat<typename T1::elem_type>& out, const Glue<T1,T2,glue_conv>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap_check<T1> A_tmp(X.A, out);
  const unwrap_check<T2> B_tmp(X.B, out);
  
  const Mat<eT>& A = A_tmp.M;
  const Mat<eT>& B = B_tmp.M;
  
  arma_debug_check
    (
    ( ((A.is_vec() == false) && (A.is_empty() == false)) || ((B.is_vec() == false) && (B.is_empty() == false)) ),
    "conv(): given object is not a vector"
    );
  
  
  const Mat<eT>& h = (A.n_elem <= B.n_elem) ? A : B;
  const Mat<eT>& x = (A.n_elem <= B.n_elem) ? B : A;
  
  
  const uword   h_n_elem = h.n_elem;
  const uword   x_n_elem = x.n_elem;
  const uword out_n_elem = h_n_elem + x_n_elem - 1;
  
  
  if( (h_n_elem == 0) || (x_n_elem == 0) )
    {
    out.reset();
    return;
    }
  
  
  (A.n_cols == 1) ? out.set_size(out_n_elem, 1) : out.set_size(1, out_n_elem);
  
  
  const eT*   h_mem = h.memptr();
  const eT*   x_mem = x.memptr();
        eT* out_mem = out.memptr();
  
  
  for(uword out_i = 0; out_i < (h_n_elem-1); ++out_i)
    {
    eT acc = eT(0);
    
    uword h_i = out_i;
    
    for(uword x_i = 0; x_i <= out_i; ++x_i, --h_i)
      {
      acc += h_mem[h_i] * x_mem[x_i];
      }
    
    out_mem[out_i] = acc;
    }
  
  
  for(uword out_i = h_n_elem-1; out_i < out_n_elem - (h_n_elem-1); ++out_i)
    {
    eT acc = eT(0);
   
    uword h_i = h_n_elem - 1;
    
    for(uword x_i = out_i - h_n_elem + 1; x_i <= out_i; ++x_i, --h_i)
      {
      acc += h_mem[h_i] * x_mem[x_i];
      }
      
    out_mem[out_i] = acc;
    }
  
  
  for(uword out_i = out_n_elem - (h_n_elem-1); out_i < out_n_elem; ++out_i)
    {
    eT acc = eT(0);
    
    uword h_i = h_n_elem - 1;
    
    for(uword x_i = out_i - h_n_elem + 1; x_i < x_n_elem; ++x_i, --h_i)
      {
      acc += h_mem[h_i] * x_mem[x_i];
      }
    
    out_mem[out_i] = acc;
    }
  
  
  }



//! @}
