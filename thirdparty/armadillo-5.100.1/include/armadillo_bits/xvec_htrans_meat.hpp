// Copyright (C) 2013 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup xvec_htrans
//! @{


template<typename eT>
inline
xvec_htrans<eT>::xvec_htrans(const eT* const in_mem, const uword in_n_rows, const uword in_n_cols)
  : mem   (in_mem             )
  , n_rows(in_n_cols          )  // deliberately swapped
  , n_cols(in_n_rows          )
  , n_elem(in_n_rows*in_n_cols)
  {
  arma_extra_debug_sigprint();
  }



template<typename eT>
inline
void
xvec_htrans<eT>::extract(Mat<eT>& out) const
  {
  arma_extra_debug_sigprint();
  
  // NOTE: this function assumes that matrix 'out' has already been set to the correct size
  
  const eT*  in_mem = mem;
        eT* out_mem = out.memptr();
  
  const uword N = n_elem;
  
  for(uword ii=0; ii < N; ++ii)
    {
    out_mem[ii] = access::alt_conj( in_mem[ii] );
    }
  }



template<typename eT>
inline
eT
xvec_htrans<eT>::operator[](const uword ii) const
  {
  return access::alt_conj( mem[ii] );
  }



template<typename eT>
inline
eT
xvec_htrans<eT>::at_alt(const uword ii) const
  {
  return access::alt_conj( mem[ii] );
  }



template<typename eT>
inline
eT
xvec_htrans<eT>::at(const uword in_row, const uword in_col) const
  {
  //return (n_rows == 1) ? access::alt_conj( mem[in_col] ) : access::alt_conj( mem[in_row] );
  
  return access::alt_conj( mem[in_row + in_col] );  // either in_row or in_col must be zero, as we're storing a vector
  }



//! @}
