// Copyright (C) 2013 Conrad Sanderson
// Copyright (C) 2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup glue_max
//! @{



template<typename eT, typename T1, typename T2>
inline
void
glue_max::apply(Mat<eT>& out, const Proxy<T1>& PA, const Proxy<T2>& PB)
  {
  arma_extra_debug_sigprint();
  
  const uword n_rows = PA.get_n_rows();
  const uword n_cols = PA.get_n_cols();
  
  arma_debug_assert_same_size(n_rows, n_cols, PB.get_n_rows(), PB.get_n_cols(), "max(): given objects do not have the same size");
  
  out.set_size(n_rows, n_cols);
  
  eT* out_mem = out.memptr();
  
  if( (Proxy<T1>::prefer_at_accessor == false) && (Proxy<T2>::prefer_at_accessor == false) )
    {
    typename Proxy<T1>::ea_type A = PA.get_ea();
    typename Proxy<T2>::ea_type B = PB.get_ea();
    
    const uword N = PA.get_n_elem();
    
    for(uword i=0; i<N; ++i)
      {
      out_mem[i] = (std::max)(A[i], B[i]);
      }
    }
  else
    {
    for(uword col=0; col < n_cols; ++col)
    for(uword row=0; row < n_rows; ++row)
      {
      *out_mem = (std::max)( PA.at(row,col), PB.at(row,col) );
      
      ++out_mem;
      }
    }
  }



template<typename T, typename T1, typename T2>
inline
void
glue_max::apply(Mat< std::complex<T> >& out, const Proxy<T1>& PA, const Proxy<T2>& PB)
  {
  arma_extra_debug_sigprint();
  
  typedef typename std::complex<T> eT;
  
  const uword n_rows = PA.get_n_rows();
  const uword n_cols = PA.get_n_cols();
  
  arma_debug_assert_same_size(n_rows, n_cols, PB.get_n_rows(), PB.get_n_cols(), "max(): given objects do not have the same size");
  
  out.set_size(n_rows, n_cols);
  
  eT* out_mem = out.memptr();
  
  if( (Proxy<T1>::prefer_at_accessor == false) && (Proxy<T2>::prefer_at_accessor == false) )
    {
    typename Proxy<T1>::ea_type A = PA.get_ea();
    typename Proxy<T2>::ea_type B = PB.get_ea();
    
    const uword N = PA.get_n_elem();
    
    for(uword i=0; i<N; ++i)
      {
      const eT A_val = A[i];
      const eT B_val = B[i];
      
      out_mem[i] = ( std::abs(A_val) > std::abs(B_val) ) ? A_val : B_val;
      }
    }
  else
    {
    for(uword col=0; col < n_cols; ++col)
    for(uword row=0; row < n_rows; ++row)
      {
      const eT A_val = PA.at(row,col);
      const eT B_val = PB.at(row,col);
      
      *out_mem = ( std::abs(A_val) > std::abs(B_val) ) ? A_val : B_val;
      
      ++out_mem;
      }
    }
  }



template<typename T1, typename T2>
inline
void
glue_max::apply(Mat<typename T1::elem_type>& out, const Glue<T1, T2, glue_max>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const Proxy<T1> PA(X.A);
  const Proxy<T2> PB(X.B);
  
  if(PA.is_alias(out) || PB.is_alias(out))
    {
    Mat<eT> tmp;
    
    glue_max::apply(tmp, PA, PB);
    
    out.steal_mem(tmp);
    }
  else
    {
    glue_max::apply(out, PA, PB);
    }
  }



//! @}
