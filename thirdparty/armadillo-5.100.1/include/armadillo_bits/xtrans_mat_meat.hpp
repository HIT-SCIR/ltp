// Copyright (C) 2014 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup xtrans_mat
//! @{


template<typename eT, bool do_conj>
inline
xtrans_mat<eT,do_conj>::xtrans_mat(const Mat<eT>& in_X)
  : X     (in_X       )
  , n_rows(in_X.n_cols)  // deliberately swapped
  , n_cols(in_X.n_rows)
  , n_elem(in_X.n_elem)
  {
  arma_extra_debug_sigprint();
  }



template<typename eT, bool do_conj>
inline
void
xtrans_mat<eT,do_conj>::extract(Mat<eT>& out) const
  {
  arma_extra_debug_sigprint();
  
  do_conj ? op_htrans::apply_mat(out, X) : op_strans::apply_mat(out, X);
  }



template<typename eT, bool do_conj>
inline
eT
xtrans_mat<eT,do_conj>::operator[](const uword ii) const
  {
  if(Y.n_elem > 0)
    {
    return Y[ii];
    }
  else
    {
    do_conj ? op_htrans::apply_mat(Y, X) : op_strans::apply_mat(Y, X);
    return Y[ii];
    }
  }



template<typename eT, bool do_conj>
inline
eT
xtrans_mat<eT,do_conj>::at_alt(const uword ii) const
  {
  return (*this).operator[](ii);
  }



template<typename eT, bool do_conj>
arma_inline
eT
xtrans_mat<eT,do_conj>::at(const uword in_row, const uword in_col) const
  {
  if(do_conj)
    {
    return access::alt_conj( X.at(in_col, in_row) ); // deliberately swapped
    }
  else
    {
    return X.at(in_col, in_row); // deliberately swapped
    }
  }



//! @}
