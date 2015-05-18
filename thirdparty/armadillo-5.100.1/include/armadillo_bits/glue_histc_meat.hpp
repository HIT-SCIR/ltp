// Copyright (C) 2012-2015 Conrad Sanderson
// Copyright (C) 2012-2015 NICTA (www.nicta.com.au)
// Copyright (C) 2012 Boris Sabanin
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



template<typename T1, typename T2>
inline
void
glue_histc::apply(Mat<uword>& out, const mtGlue<uword,T1,T2,glue_histc>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword dim = in.aux_uword;
  
  const unwrap_check_mixed<T1> tmp1(in.A, out);
  const unwrap_check_mixed<T2> tmp2(in.B, out);
  
  const Mat<eT>& X = tmp1.M;
  const Mat<eT>& E = tmp2.M;
  
  arma_debug_check
    (
    ((E.is_vec() == false) && (E.is_empty() == false)),
    "histc(): parameter 'edges' must be a vector"
    );
  
  arma_debug_check
    (
    (dim > 1),
    "histc(): parameter 'dim' must be 0 or 1"
    );
  
  const uword X_n_elem = X.n_elem;
  const uword X_n_rows = X.n_rows;
  const uword X_n_cols = X.n_cols;
  
  const uword E_n_elem = E.n_elem;
  
  if( E_n_elem == 0 )
    {
    out.reset();
    return;
    }
  
  
  // for vectors we are currently ignoring the "dim" parameter
  
  uword out_n_rows = 0;
  uword out_n_cols = 0;
  
  if( (X.vec_state == 0) && (X.n_elem == 1u) )
    {
    if(out.vec_state == 1u)
      {
      out_n_rows = E_n_elem;
      out_n_cols = 1;
      }
    else
      {
      out_n_rows = 1;
      out_n_cols = E_n_elem;
      }
    }
  else
  if( (X.vec_state > 0) || X.is_vec() )
    {
    if(X.vec_state == 2u)
      {
      out_n_rows = 1;
      out_n_cols = E_n_elem;
      }
    else
    if(X.vec_state == 1u)
      {
      out_n_rows = E_n_elem;
      out_n_cols = 1;
      }
    else
    if(X.is_rowvec())
      {
      out_n_rows = 1;
      out_n_cols = E_n_elem;
      }
    else
    if(X.is_colvec())
      {
      out_n_rows = E_n_elem;
      out_n_cols = 1;
      }
    }
  else
    {
    if(dim == 0)
      {
      out_n_rows = E_n_elem;
      out_n_cols = X_n_cols;
      }
    else
    if(dim == 1)
      {
      out_n_rows = X_n_rows;
      out_n_cols = E_n_elem;
      }
    }
  
  out.zeros(out_n_rows, out_n_cols);
  
  const eT* E_mem = E.memptr();

  if( (X.vec_state > 0) || X.is_vec() )
    {
          uword* out_mem = out.memptr();
    const eT*    X_mem   = X.memptr();
    
    for(uword j=0; j<X_n_elem; ++j)
      {
      const eT val = X_mem[j];
      
      for(uword i=0; i<E_n_elem-1; ++i)
        {
        if( (E_mem[i] <= val) && (val < E_mem[i+1]) )
          {
          out_mem[i]++;
          break;
          }
        else
        if(val == E_mem[E_n_elem-1])
          {
          // in general, the above == operation doesn't make sense for floating point values (due to precision issues),
          // but is included for compatibility with Matlab and Octave.
          // Matlab folks must have been smoking something strong.
          out_mem[E_n_elem-1]++;
          break;
          }
        }
      }
    }
  else
  if(dim == 0)
    {
    for(uword col=0; col<X_n_cols; ++col)
      {
            uword* out_coldata = out.colptr(col);
      const eT*    X_coldata   = X.colptr(col);
      
      for(uword row=0; row<X_n_rows; ++row)
        {
        const eT val = X_coldata[row];
        
        for(uword i=0; i<E_n_elem-1; ++i)
          {
          if( (E_mem[i] <= val) && (val < E_mem[i+1]) )
            {
            out_coldata[i]++;
            break;
            }
          else
          if(val == E_mem[E_n_elem-1])
            {
            out_coldata[E_n_elem-1]++;
            break;
            }
          }
        }
      }
    }
  else
  if(dim == 1)
    {
    for(uword row=0; row<X_n_rows; ++row)
      {
      for(uword col=0; col<X_n_cols; ++col)
        {
        const eT val = X.at(row,col);
        
        for(uword i=0; i<E_n_elem-1; ++i)
          {
          if( (E_mem[i] <= val) && (val < E_mem[i+1]) )
            {
            out.at(row,i)++;
            break;
            }
          else
          if(val == E_mem[E_n_elem-1])
            {
            out.at(row,E_n_elem-1)++;
            break;
            }
          }
        }
      }
    }
  }
