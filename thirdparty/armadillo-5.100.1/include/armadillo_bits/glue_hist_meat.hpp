// Copyright (C) 2012-2015 Conrad Sanderson
// Copyright (C) 2012-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



template<typename T1, typename T2>
inline
void
glue_hist::apply(Mat<uword>& out, const mtGlue<uword,T1,T2,glue_hist>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword dim = in.aux_uword;
  
  const unwrap_check_mixed<T1> tmp1(in.A, out);
  const unwrap_check_mixed<T2> tmp2(in.B, out);
  
  const Mat<eT>& X = tmp1.M;
  const Mat<eT>& C = tmp2.M;
  
  
  arma_debug_check
    (
    ((C.is_vec() == false) && (C.is_empty() == false)),
    "hist(): parameter 'centers' must be a vector"
    );
  
  arma_debug_check
    (
    (dim > 1),
    "hist(): parameter 'dim' must be 0 or 1"
    );
  
  const uword X_n_rows = X.n_rows;
  const uword X_n_cols = X.n_cols;
  const uword X_n_elem = X.n_elem;
  
  const uword C_n_elem = C.n_elem;
  
  if( C_n_elem == 0 )
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
      out_n_rows = C_n_elem;
      out_n_cols = 1;
      }
    else
      {
      out_n_rows = 1;
      out_n_cols = C_n_elem;
      }
    }
  else
  if( (X.vec_state > 0) || X.is_vec() )
    {
    if(X.vec_state == 2u)
      {
      out_n_rows = 1;
      out_n_cols = C_n_elem;
      }
    else
    if(X.vec_state == 1u)
      {
      out_n_rows = C_n_elem;
      out_n_cols = 1;
      }
    else
    if(X.is_rowvec())
      {
      out_n_rows = 1;
      out_n_cols = C_n_elem;
      }
    else
    if(X.is_colvec())
      {
      out_n_rows = C_n_elem;
      out_n_cols = 1;
      }
    }
  else
    {
    if(dim == 0)
      {
      out_n_rows = C_n_elem;
      out_n_cols = X_n_cols;
      }
    else
    if(dim == 1)
      {
      out_n_rows = X_n_rows;
      out_n_cols = C_n_elem;
      }
    }
  
  out.zeros(out_n_rows, out_n_cols);
  
  
  const eT* C_mem    = C.memptr();
  const eT  center_0 = C_mem[0];
  
  if( (X.vec_state > 0) || X.is_vec() )
    {
    const eT*    X_mem   = X.memptr();
          uword* out_mem = out.memptr();
    
    for(uword i=0; i < X_n_elem; ++i)
      {
      const eT val = X_mem[i];
      
      if(is_finite(val))
        {
        eT    opt_dist  = (val >= center_0) ? (val - center_0) : (center_0 - val);
        uword opt_index = 0;
        
        for(uword j=1; j < C_n_elem; ++j)
          {
          const eT center = C_mem[j];
          const eT dist   = (val >= center) ? (val - center) : (center - val);
          
          if(dist < opt_dist)
            {
            opt_dist  = dist;
            opt_index = j;
            }
          else
            {
            break;
            }
          }
        
        out_mem[opt_index]++;
        }
      else
        {
        // -inf
        if(val < eT(0)) { out_mem[0]++; }
        
        // +inf
        if(val > eT(0)) { out_mem[C_n_elem-1]++; }
        
        // ignore NaN
        }
      }
    }
  else
    {
    if(dim == 0)
      {
      for(uword col=0; col < X_n_cols; ++col)
        {
        const eT*    X_coldata   = X.colptr(col);
              uword* out_coldata = out.colptr(col);
        
        for(uword row=0; row < X_n_rows; ++row)
          {
          const eT val = X_coldata[row];
          
          if(arma_isfinite(val))
            {
            eT    opt_dist  = (center_0 >= val) ? (center_0 - val) : (val - center_0);
            uword opt_index = 0;
            
            for(uword j=1; j < C_n_elem; ++j)
              {
              const eT center = C_mem[j];
              const eT dist   = (center >= val) ? (center - val) : (val - center);
              
              if(dist < opt_dist)
                {
                opt_dist  = dist;
                opt_index = j;
                }
              else
                {
                break;
                }
              }
            
            out_coldata[opt_index]++;
            }
          else
            {
            // -inf
            if(val < eT(0)) { out_coldata[0]++; }
            
            // +inf
            if(val > eT(0)) { out_coldata[C_n_elem-1]++; }
            
            // ignore NaN
            }
          }
        }
      }
    else
    if(dim == 1)
      {
      for(uword row=0; row < X_n_rows; ++row)
        {
        for(uword col=0; col < X_n_cols; ++col)
          {
          const eT val = X.at(row,col);
          
          if(arma_isfinite(val))
            {
            eT    opt_dist  = (center_0 >= val) ? (center_0 - val) : (val - center_0);
            uword opt_index = 0;
            
            for(uword j=1; j < C_n_elem; ++j)
              {
              const eT center = C_mem[j];
              const eT dist   = (center >= val) ? (center - val) : (val - center);
              
              if(dist < opt_dist)
                {
                opt_dist  = dist;
                opt_index = j;
                }
              else
                {
                break;
                }
              }
            
            out.at(row,opt_index)++;
            }
          else
            {
            // -inf
            if(val < eT(0)) { out.at(row,0)++; }
            
            // +inf
            if(val > eT(0)) { out.at(row,C_n_elem-1)++; }
            
            // ignore NaN
            }
          }
        }
      }
    }
  }
