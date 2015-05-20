// Copyright (C) 2012-2014 Ryan Curtin
// Copyright (C) 2012-2014 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup spglue_plus
//! @{



template<typename T1, typename T2>
arma_hot
inline
void
spglue_plus::apply(SpMat<typename T1::elem_type>& out, const SpGlue<T1,T2,spglue_plus>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const SpProxy<T1> pa(X.A);
  const SpProxy<T2> pb(X.B);
  
  const bool is_alias = pa.is_alias(out) || pb.is_alias(out);
  
  if(is_alias == false)
    {
    spglue_plus::apply_noalias(out, pa, pb);
    }
  else
    {
    SpMat<eT> tmp;
    spglue_plus::apply_noalias(tmp, pa, pb);
    
    out.steal_mem(tmp);
    }
  }



template<typename eT, typename T1, typename T2>
arma_hot
inline
void
spglue_plus::apply_noalias(SpMat<eT>& out, const SpProxy<T1>& pa, const SpProxy<T2>& pb)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(pa.get_n_rows(), pa.get_n_cols(), pb.get_n_rows(), pb.get_n_cols(), "addition");
  
  if( (pa.get_n_nonzero() != 0) && (pb.get_n_nonzero() != 0) )
    {
    out.set_size(pa.get_n_rows(), pa.get_n_cols());
    
    // Resize memory to correct size.
    out.mem_resize(n_unique(pa, pb, op_n_unique_add()));
    
    // Now iterate across both matrices.
    typename SpProxy<T1>::const_iterator_type x_it = pa.begin();
    typename SpProxy<T2>::const_iterator_type y_it = pb.begin();
    
    typename SpProxy<T1>::const_iterator_type x_end = pa.end();
    typename SpProxy<T2>::const_iterator_type y_end = pb.end();
    
    uword cur_val = 0;
    while( (x_it != x_end) || (y_it != y_end) )
      {
      if(x_it == y_it)
        {
        const eT val = (*x_it) + (*y_it);
        
        if(val != eT(0))
          {
          access::rw(out.values[cur_val]) = val;
          access::rw(out.row_indices[cur_val]) = x_it.row();
          ++access::rw(out.col_ptrs[x_it.col() + 1]);
          ++cur_val;
          }
        
        ++x_it;
        ++y_it;
        }
      else
        {
        const uword x_it_row = x_it.row();
        const uword x_it_col = x_it.col();
        
        const uword y_it_row = y_it.row();
        const uword y_it_col = y_it.col();
        
        if((x_it_col < y_it_col) || ((x_it_col == y_it_col) && (x_it_row < y_it_row))) // if y is closer to the end
          {
          const eT val = (*x_it);
          
          if(val != eT(0))
            {
            access::rw(out.values[cur_val]) = val;
            access::rw(out.row_indices[cur_val]) = x_it_row;
            ++access::rw(out.col_ptrs[x_it_col + 1]);
            ++cur_val;
            }
          
          ++x_it;
          }
        else
          {
          const eT val = (*y_it);
          
          if(val != eT(0))
            {
            access::rw(out.values[cur_val]) = val;
            access::rw(out.row_indices[cur_val]) = y_it_row;
            ++access::rw(out.col_ptrs[y_it_col + 1]);
            ++cur_val;
            }
          
          ++y_it;
          }
        }
      }
    
    const uword out_n_cols = out.n_cols;
    
    uword* col_ptrs = access::rwp(out.col_ptrs);
    
    // Fix column pointers to be cumulative.
    for(uword c = 1; c <= out_n_cols; ++c)
      {
      col_ptrs[c] += col_ptrs[c - 1];
      }
    }
  else
    {
    if(pa.get_n_nonzero() == 0)
      {
      out = pb.Q;
      return;
      }
    
    if(pb.get_n_nonzero() == 0)
      {
      out = pa.Q;
      return;
      }
    }
  }



//
//
// spglue_plus2: scalar*(A + B)



template<typename T1, typename T2>
arma_hot
inline
void
spglue_plus2::apply(SpMat<typename T1::elem_type>& out, const SpGlue<T1,T2,spglue_plus2>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const SpProxy<T1> pa(X.A);
  const SpProxy<T2> pb(X.B);
  
  const bool is_alias = pa.is_alias(out) || pb.is_alias(out);
  
  if(is_alias == false)
    {
    spglue_plus::apply_noalias(out, pa, pb);
    }
  else
    {
    SpMat<eT> tmp;
    spglue_plus::apply_noalias(tmp, pa, pb);
    
    out.steal_mem(tmp);
    }
  
  out *= X.aux;
  }



//! @}
