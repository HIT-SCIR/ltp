// Copyright (C) 2013-2015 Conrad Sanderson
// Copyright (C) 2013-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup op_vectorise
//! @{



template<typename T1>
inline
void
op_vectorise_col::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_vectorise_col>& in)
  {
  arma_extra_debug_sigprint();
  
  const Proxy<T1> P(in.m);
  
  op_vectorise_col::apply_proxy(out, P);
  }



template<typename T1>
inline
void
op_vectorise_col::apply_proxy(Mat<typename T1::elem_type>& out, const Proxy<T1>& P)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  if(P.is_alias(out) == false)
    {
    const uword N = P.get_n_elem();
    
    out.set_size(N, 1);
      
    if(is_Mat<typename Proxy<T1>::stored_type>::value == true)
      {
      const unwrap<typename Proxy<T1>::stored_type> tmp(P.Q);
      
      arrayops::copy(out.memptr(), tmp.M.memptr(), N);
      }
    else
      {
      eT* outmem = out.memptr();
      
      if(Proxy<T1>::prefer_at_accessor == false)
        {
        // TODO: add handling of aligned access ?
        
        typename Proxy<T1>::ea_type A = P.get_ea();
        
        uword i,j;
        
        for(i=0, j=1; j < N; i+=2, j+=2)
          {
          const eT tmp_i = A[i];
          const eT tmp_j = A[j];
          
          outmem[i] = tmp_i;
          outmem[j] = tmp_j;
          }
        
        if(i < N)
          {
          outmem[i] = A[i];
          }
        }
      else
        {
        const uword n_rows = P.get_n_rows();
        const uword n_cols = P.get_n_cols();
        
        if(n_rows == 1)
          {
          for(uword i=0; i < n_cols; ++i)
            {
            outmem[i] = P.at(0,i);
            }
          }
        else
          {
          for(uword col=0; col < n_cols; ++col)
          for(uword row=0; row < n_rows; ++row)
            {
            *outmem = P.at(row,col);
            outmem++;
            }
          }
        }
      }
    }
  else  // we have aliasing
    {
    arma_extra_debug_print("op_vectorise_col::apply(): aliasing detected");
    
    if( (is_Mat<typename Proxy<T1>::stored_type>::value == true) && (Proxy<T1>::fake_mat == false) )
      {
      out.set_size(out.n_elem, 1);  // set_size() doesn't destroy data as long as the number of elements in the matrix remains the same
      }
    else
      {
      Mat<eT> tmp;
      
      op_vectorise_col::apply_proxy(tmp, P);
      
      out.steal_mem(tmp);
      }
    }
  }



template<typename T1>
inline
void
op_vectorise_row::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_vectorise_row>& in)
  {
  arma_extra_debug_sigprint();
  
  const Proxy<T1> P(in.m);
  
  op_vectorise_row::apply_proxy(out, P);
  }



template<typename T1>
inline
void
op_vectorise_row::apply_proxy(Mat<typename T1::elem_type>& out, const Proxy<T1>& P)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  if(P.is_alias(out) == false)
    {
    out.set_size( 1, P.get_n_elem() );
    
    eT* outmem = out.memptr();
    
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    for(uword row=0; row < n_rows; ++row)
      {
      uword i,j;
      
      for(i=0, j=1; j < n_cols; i+=2, j+=2)
        {
        const eT tmp_i = P.at(row,i);
        const eT tmp_j = P.at(row,j);
        
        *outmem = tmp_i; outmem++;
        *outmem = tmp_j; outmem++;
        }
      
      if(i < n_cols)
        {
        *outmem = P.at(row,i); outmem++;
        }
      }
    }
  else  // we have aliasing
    {
    arma_extra_debug_print("op_vectorise_row::apply(): aliasing detected");
    
    Mat<eT> tmp;
    
    op_vectorise_row::apply_proxy(tmp, P);
    
    out.steal_mem(tmp);
    }
  }



template<typename T1>
inline
void
op_vectorise_all::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_vectorise_all>& in)
  {
  arma_extra_debug_sigprint();
  
  const Proxy<T1> P(in.m);
  
  const uword dim = in.aux_uword_a;
  
  if(dim == 0)
    {
    op_vectorise_col::apply_proxy(out, P);
    }
  else
    {
    op_vectorise_row::apply_proxy(out, P);
    }
  }



//



template<typename T1>
inline
void
op_vectorise_cube_col::apply(Mat<typename T1::elem_type>& out, const BaseCube<typename T1::elem_type, T1>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  ProxyCube<T1> P(in.get_ref());
  
  const uword N = P.get_n_elem();
  
  out.set_size(N, 1);
  
  if(is_Cube<typename ProxyCube<T1>::stored_type>::value == true)
    {
    const unwrap_cube<typename ProxyCube<T1>::stored_type> tmp(P.Q);
    
    arrayops::copy(out.memptr(), tmp.M.memptr(), N);
    }
  else
    {
    eT* outmem = out.memptr();
    
    if(ProxyCube<T1>::prefer_at_accessor == false)
      {
      typename ProxyCube<T1>::ea_type A = P.get_ea();
      
      uword i,j;
      
      for(i=0, j=1; j < N; i+=2, j+=2)
        {
        const eT tmp_i = A[i];
        const eT tmp_j = A[j];
        
        outmem[i] = tmp_i;
        outmem[j] = tmp_j;
        }
      
      if(i < N)
        {
        outmem[i] = A[i];
        }
      }
    else
      {
      const uword n_rows   = P.get_n_rows();
      const uword n_cols   = P.get_n_cols();
      const uword n_slices = P.get_n_slices();
      
      for(uword slice=0; slice < n_slices; ++slice)
      for(uword   col=0;   col < n_cols;   ++col  )
      for(uword   row=0;   row < n_rows;   ++row  )
        {
        *outmem = P.at(row,col,slice);
        outmem++;
        }
      }
    }
  }



//! @}
