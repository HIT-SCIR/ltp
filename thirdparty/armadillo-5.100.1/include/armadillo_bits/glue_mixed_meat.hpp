// Copyright (C) 2009-2013 Conrad Sanderson
// Copyright (C) 2009-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup glue_mixed
//! @{



//! matrix multiplication with different element types
template<typename T1, typename T2>
inline
void
glue_mixed_times::apply(Mat<typename eT_promoter<T1,T2>::eT>& out, const mtGlue<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_times>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  const unwrap_check_mixed<T1> tmp1(X.A, out);
  const unwrap_check_mixed<T2> tmp2(X.B, out);
  
  const Mat<eT1>& A = tmp1.M;
  const Mat<eT2>& B = tmp2.M;
  
  arma_debug_assert_mul_size(A, B, "matrix multiplication");
  
  out.set_size(A.n_rows, B.n_cols);
  
  gemm_mixed<>::apply(out, A, B);
  }



//! matrix addition with different element types
template<typename T1, typename T2>
inline
void
glue_mixed_plus::apply(Mat<typename eT_promoter<T1,T2>::eT>& out, const mtGlue<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_plus>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  const Proxy<T1> A(X.A);
  const Proxy<T2> B(X.B);
  
  arma_debug_assert_same_size(A, B, "addition");
  
  const uword n_rows = A.get_n_rows();
  const uword n_cols = A.get_n_cols();
  
  out.set_size(n_rows, n_cols);
  
        out_eT* out_mem = out.memptr();
  const uword   n_elem  = out.n_elem;
    
  const bool prefer_at_accessor = (Proxy<T1>::prefer_at_accessor || Proxy<T2>::prefer_at_accessor);
  
  if(prefer_at_accessor == false)
    {
    typename Proxy<T1>::ea_type AA = A.get_ea();
    typename Proxy<T2>::ea_type BB = B.get_ea();
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      for(uword i=0; i<n_elem; ++i)
        {
        out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) + upgrade_val<eT1,eT2>::apply(BB[i]);
        }
      }
    else
      {
      for(uword i=0; i<n_elem; ++i)
        {
        out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) + upgrade_val<eT1,eT2>::apply(BB[i]);
        }
      }
    }
  else
    {
    uword i = 0;
    
    for(uword col=0; col < n_cols; ++col)
    for(uword row=0; row < n_rows; ++row)
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(A.at(row,col)) + upgrade_val<eT1,eT2>::apply(B.at(row,col));
      ++i;
      }
    }
  }



//! matrix subtraction with different element types
template<typename T1, typename T2>
inline
void
glue_mixed_minus::apply(Mat<typename eT_promoter<T1,T2>::eT>& out, const mtGlue<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_minus>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  const Proxy<T1> A(X.A);
  const Proxy<T2> B(X.B);
  
  arma_debug_assert_same_size(A, B, "subtraction");
  
  const uword n_rows = A.get_n_rows();
  const uword n_cols = A.get_n_cols();
  
  out.set_size(n_rows, n_cols);
  
        out_eT* out_mem = out.memptr();
  const uword   n_elem  = out.n_elem;
    
  const bool prefer_at_accessor = (Proxy<T1>::prefer_at_accessor || Proxy<T2>::prefer_at_accessor);
  
  if(prefer_at_accessor == false)
    {
    typename Proxy<T1>::ea_type AA = A.get_ea();
    typename Proxy<T2>::ea_type BB = B.get_ea();
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      for(uword i=0; i<n_elem; ++i)
        {
        out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) - upgrade_val<eT1,eT2>::apply(BB[i]);
        }
      }
    else
      {
      for(uword i=0; i<n_elem; ++i)
        {
        out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) - upgrade_val<eT1,eT2>::apply(BB[i]);
        }
      }
    }
  else
    {
    uword i = 0;
    
    for(uword col=0; col < n_cols; ++col)
    for(uword row=0; row < n_rows; ++row)
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(A.at(row,col)) - upgrade_val<eT1,eT2>::apply(B.at(row,col));
      ++i;
      }
    }
  }



//! element-wise matrix division with different element types
template<typename T1, typename T2>
inline
void
glue_mixed_div::apply(Mat<typename eT_promoter<T1,T2>::eT>& out, const mtGlue<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_div>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  const Proxy<T1> A(X.A);
  const Proxy<T2> B(X.B);
  
  arma_debug_assert_same_size(A, B, "element-wise division");
  
  const uword n_rows = A.get_n_rows();
  const uword n_cols = A.get_n_cols();
  
  out.set_size(n_rows, n_cols);
  
        out_eT* out_mem = out.memptr();
  const uword   n_elem  = out.n_elem;
    
  const bool prefer_at_accessor = (Proxy<T1>::prefer_at_accessor || Proxy<T2>::prefer_at_accessor);
  
  if(prefer_at_accessor == false)
    {
    typename Proxy<T1>::ea_type AA = A.get_ea();
    typename Proxy<T2>::ea_type BB = B.get_ea();
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      for(uword i=0; i<n_elem; ++i)
        {
        out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) / upgrade_val<eT1,eT2>::apply(BB[i]);
        }
      }
    else
      {
      for(uword i=0; i<n_elem; ++i)
        {
        out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) / upgrade_val<eT1,eT2>::apply(BB[i]);
        }
      }
    }
  else
    {
    uword i = 0;
    
    for(uword col=0; col < n_cols; ++col)
    for(uword row=0; row < n_rows; ++row)
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(A.at(row,col)) / upgrade_val<eT1,eT2>::apply(B.at(row,col));
      ++i;
      }
    }
  }



//! element-wise matrix multiplication with different element types
template<typename T1, typename T2>
inline
void
glue_mixed_schur::apply(Mat<typename eT_promoter<T1,T2>::eT>& out, const mtGlue<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_schur>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  const Proxy<T1> A(X.A);
  const Proxy<T2> B(X.B);
  
  arma_debug_assert_same_size(A, B, "element-wise multiplication");
  
  const uword n_rows = A.get_n_rows();
  const uword n_cols = A.get_n_cols();
  
  out.set_size(n_rows, n_cols);
  
        out_eT* out_mem = out.memptr();
  const uword   n_elem  = out.n_elem;
    
  const bool prefer_at_accessor = (Proxy<T1>::prefer_at_accessor || Proxy<T2>::prefer_at_accessor);
  
  if(prefer_at_accessor == false)
    {
    typename Proxy<T1>::ea_type AA = A.get_ea();
    typename Proxy<T2>::ea_type BB = B.get_ea();
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      for(uword i=0; i<n_elem; ++i)
        {
        out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) * upgrade_val<eT1,eT2>::apply(BB[i]);
        }
      }
    else
      {
      for(uword i=0; i<n_elem; ++i)
        {
        out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) * upgrade_val<eT1,eT2>::apply(BB[i]);
        }
      }
    }
  else
    {
    uword i = 0;
    
    for(uword col=0; col < n_cols; ++col)
    for(uword row=0; row < n_rows; ++row)
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(A.at(row,col)) * upgrade_val<eT1,eT2>::apply(B.at(row,col));
      ++i;
      }
    }
  }



//
//
//



//! cube addition with different element types
template<typename T1, typename T2>
inline
void
glue_mixed_plus::apply(Cube<typename eT_promoter<T1,T2>::eT>& out, const mtGlueCube<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_plus>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  const ProxyCube<T1> A(X.A);
  const ProxyCube<T2> B(X.B);
  
  arma_debug_assert_same_size(A, B, "addition");
  
  const uword n_rows   = A.get_n_rows();
  const uword n_cols   = A.get_n_cols();
  const uword n_slices = A.get_n_slices();

  out.set_size(n_rows, n_cols, n_slices);
  
        out_eT* out_mem = out.memptr();
  const uword    n_elem = out.n_elem;
  
  const bool prefer_at_accessor = (ProxyCube<T1>::prefer_at_accessor || ProxyCube<T2>::prefer_at_accessor);
  
  if(prefer_at_accessor == false)
    {
    typename ProxyCube<T1>::ea_type AA = A.get_ea();
    typename ProxyCube<T2>::ea_type BB = B.get_ea();
    
    for(uword i=0; i<n_elem; ++i)
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) + upgrade_val<eT1,eT2>::apply(BB[i]);
      }
    }
  else
    {
    uword i = 0;
    
    for(uword slice = 0; slice < n_slices; ++slice)
    for(uword col   = 0; col   < n_cols;   ++col  )
    for(uword row   = 0; row   < n_rows;   ++row  )
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(A.at(row,col,slice)) + upgrade_val<eT1,eT2>::apply(B.at(row,col,slice));
      ++i;
      }
    }
  }



//! cube subtraction with different element types
template<typename T1, typename T2>
inline
void
glue_mixed_minus::apply(Cube<typename eT_promoter<T1,T2>::eT>& out, const mtGlueCube<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_minus>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  const ProxyCube<T1> A(X.A);
  const ProxyCube<T2> B(X.B);
  
  arma_debug_assert_same_size(A, B, "subtraction");
  
  const uword n_rows   = A.get_n_rows();
  const uword n_cols   = A.get_n_cols();
  const uword n_slices = A.get_n_slices();

  out.set_size(n_rows, n_cols, n_slices);
  
        out_eT* out_mem = out.memptr();
  const uword    n_elem = out.n_elem;
  
  const bool prefer_at_accessor = (ProxyCube<T1>::prefer_at_accessor || ProxyCube<T2>::prefer_at_accessor);
  
  if(prefer_at_accessor == false)
    {
    typename ProxyCube<T1>::ea_type AA = A.get_ea();
    typename ProxyCube<T2>::ea_type BB = B.get_ea();
    
    for(uword i=0; i<n_elem; ++i)
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) - upgrade_val<eT1,eT2>::apply(BB[i]);
      }
    }
  else
    {
    uword i = 0;
    
    for(uword slice = 0; slice < n_slices; ++slice)
    for(uword col   = 0; col   < n_cols;   ++col  )
    for(uword row   = 0; row   < n_rows;   ++row  )
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(A.at(row,col,slice)) - upgrade_val<eT1,eT2>::apply(B.at(row,col,slice));
      ++i;
      }
    }
  }



//! element-wise cube division with different element types
template<typename T1, typename T2>
inline
void
glue_mixed_div::apply(Cube<typename eT_promoter<T1,T2>::eT>& out, const mtGlueCube<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_div>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  const ProxyCube<T1> A(X.A);
  const ProxyCube<T2> B(X.B);
  
  arma_debug_assert_same_size(A, B, "element-wise division");
  
  const uword n_rows   = A.get_n_rows();
  const uword n_cols   = A.get_n_cols();
  const uword n_slices = A.get_n_slices();

  out.set_size(n_rows, n_cols, n_slices);
  
        out_eT* out_mem = out.memptr();
  const uword    n_elem = out.n_elem;
  
  const bool prefer_at_accessor = (ProxyCube<T1>::prefer_at_accessor || ProxyCube<T2>::prefer_at_accessor);
  
  if(prefer_at_accessor == false)
    {
    typename ProxyCube<T1>::ea_type AA = A.get_ea();
    typename ProxyCube<T2>::ea_type BB = B.get_ea();
    
    for(uword i=0; i<n_elem; ++i)
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) / upgrade_val<eT1,eT2>::apply(BB[i]);
      }
    }
  else
    {
    uword i = 0;
    
    for(uword slice = 0; slice < n_slices; ++slice)
    for(uword col   = 0; col   < n_cols;   ++col  )
    for(uword row   = 0; row   < n_rows;   ++row  )
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(A.at(row,col,slice)) / upgrade_val<eT1,eT2>::apply(B.at(row,col,slice));
      ++i;
      }
    }
  }



//! element-wise cube multiplication with different element types
template<typename T1, typename T2>
inline
void
glue_mixed_schur::apply(Cube<typename eT_promoter<T1,T2>::eT>& out, const mtGlueCube<typename eT_promoter<T1,T2>::eT, T1, T2, glue_mixed_schur>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  typedef typename promote_type<eT1,eT2>::result out_eT;
  
  promote_type<eT1,eT2>::check();
  
  const ProxyCube<T1> A(X.A);
  const ProxyCube<T2> B(X.B);
  
  arma_debug_assert_same_size(A, B, "element-wise multiplication");
  
  const uword n_rows   = A.get_n_rows();
  const uword n_cols   = A.get_n_cols();
  const uword n_slices = A.get_n_slices();

  out.set_size(n_rows, n_cols, n_slices);
  
        out_eT* out_mem = out.memptr();
  const uword    n_elem = out.n_elem;
  
  const bool prefer_at_accessor = (ProxyCube<T1>::prefer_at_accessor || ProxyCube<T2>::prefer_at_accessor);
  
  if(prefer_at_accessor == false)
    {
    typename ProxyCube<T1>::ea_type AA = A.get_ea();
    typename ProxyCube<T2>::ea_type BB = B.get_ea();
    
    for(uword i=0; i<n_elem; ++i)
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(AA[i]) * upgrade_val<eT1,eT2>::apply(BB[i]);
      }
    }
  else
    {
    uword i = 0;
    
    for(uword slice = 0; slice < n_slices; ++slice)
    for(uword col   = 0; col   < n_cols;   ++col  )
    for(uword row   = 0; row   < n_rows;   ++row  )
      {
      out_mem[i] = upgrade_val<eT1,eT2>::apply(A.at(row,col,slice)) * upgrade_val<eT1,eT2>::apply(B.at(row,col,slice));
      ++i;
      }
    }
  }



//! @}
