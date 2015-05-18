// Copyright (C) 2011-2013 Conrad Sanderson
// Copyright (C) 2011-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup Gen
//! @{



template<typename eT, typename gen_type>
arma_inline
GenCube<eT, gen_type>::GenCube(const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  : n_rows  (in_n_rows  )
  , n_cols  (in_n_cols  )
  , n_slices(in_n_slices)
  {
  arma_extra_debug_sigprint();
  }



template<typename eT, typename gen_type>
arma_inline
GenCube<eT, gen_type>::~GenCube()
  {
  arma_extra_debug_sigprint();
  }



template<typename eT, typename gen_type>
arma_inline
eT
GenCube<eT, gen_type>::generate()
  {
       if(is_same_type<gen_type, gen_ones_full>::yes) { return eT(1);                     }
  else if(is_same_type<gen_type, gen_zeros    >::yes) { return eT(0);                     }
  else if(is_same_type<gen_type, gen_randu    >::yes) { return eT(arma_rng::randu<eT>()); }
  else if(is_same_type<gen_type, gen_randn    >::yes) { return eT(arma_rng::randn<eT>()); }
  else                                                { return eT();                      }
  }



template<typename eT, typename gen_type>
arma_inline
eT
GenCube<eT, gen_type>::operator[](const uword) const
  {
  return GenCube<eT, gen_type>::generate();
  }



template<typename eT, typename gen_type>
arma_inline
eT
GenCube<eT, gen_type>::at(const uword, const uword, const uword) const
  {
  return GenCube<eT, gen_type>::generate();
  }



template<typename eT, typename gen_type>
arma_inline
eT
GenCube<eT, gen_type>::at_alt(const uword) const
  {
  return GenCube<eT, gen_type>::generate();
  }



template<typename eT, typename gen_type>
inline
void
GenCube<eT, gen_type>::apply(Cube<eT>& out) const
  {
  arma_extra_debug_sigprint();
  
  // NOTE: we're assuming that the cube has already been set to the correct size;
  // this is done by either the Cube contructor or operator=()
  
       if(is_same_type<gen_type, gen_ones_full>::yes) { out.ones();  }
  else if(is_same_type<gen_type, gen_zeros    >::yes) { out.zeros(); }
  else if(is_same_type<gen_type, gen_randu    >::yes) { out.randu(); }
  else if(is_same_type<gen_type, gen_randn    >::yes) { out.randn(); }
  }



template<typename eT, typename gen_type>
inline
void
GenCube<eT, gen_type>::apply_inplace_plus(Cube<eT>& out) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, out.n_slices, n_rows, n_cols, n_slices, "addition");
  
  
        eT*   out_mem = out.memptr();
  const uword n_elem  = out.n_elem;
  
  uword i,j;
  
  for(i=0, j=1; j<n_elem; i+=2, j+=2)
    {
    const eT tmp_i = GenCube<eT, gen_type>::generate();
    const eT tmp_j = GenCube<eT, gen_type>::generate();
    
    out_mem[i] += tmp_i;
    out_mem[j] += tmp_j;
    }
  
  if(i < n_elem)
    {
    out_mem[i] += GenCube<eT, gen_type>::generate();
    }
  }




template<typename eT, typename gen_type>
inline
void
GenCube<eT, gen_type>::apply_inplace_minus(Cube<eT>& out) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, out.n_slices, n_rows, n_cols, n_slices, "subtraction");
  
  
        eT*   out_mem = out.memptr();
  const uword n_elem  = out.n_elem;
  
  uword i,j;
  
  for(i=0, j=1; j<n_elem; i+=2, j+=2)
    {
    const eT tmp_i = GenCube<eT, gen_type>::generate();
    const eT tmp_j = GenCube<eT, gen_type>::generate();
    
    out_mem[i] -= tmp_i;
    out_mem[j] -= tmp_j;
    }
  
  if(i < n_elem)
    {
    out_mem[i] -= GenCube<eT, gen_type>::generate();
    }
  }




template<typename eT, typename gen_type>
inline
void
GenCube<eT, gen_type>::apply_inplace_schur(Cube<eT>& out) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, out.n_slices, n_rows, n_cols, n_slices, "element-wise multiplication");
  
  
        eT*   out_mem = out.memptr();
  const uword n_elem  = out.n_elem;
  
  uword i,j;
  
  for(i=0, j=1; j<n_elem; i+=2, j+=2)
    {
    const eT tmp_i = GenCube<eT, gen_type>::generate();
    const eT tmp_j = GenCube<eT, gen_type>::generate();
    
    out_mem[i] *= tmp_i;
    out_mem[j] *= tmp_j;
    }
  
  if(i < n_elem)
    {
    out_mem[i] *= GenCube<eT, gen_type>::generate();
    }
  }




template<typename eT, typename gen_type>
inline
void
GenCube<eT, gen_type>::apply_inplace_div(Cube<eT>& out) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, out.n_slices, n_rows, n_cols, n_slices, "element-wise division");
  
  
        eT*   out_mem = out.memptr();
  const uword n_elem  = out.n_elem;
  
  uword i,j;
  
  for(i=0, j=1; j<n_elem; i+=2, j+=2)
    {
    const eT tmp_i = GenCube<eT, gen_type>::generate();
    const eT tmp_j = GenCube<eT, gen_type>::generate();
    
    out_mem[i] /= tmp_i;
    out_mem[j] /= tmp_j;
    }
  
  if(i < n_elem)
    {
    out_mem[i] /= GenCube<eT, gen_type>::generate();
    }
  }




//! @}
