// Copyright (C) 2012 Conrad Sanderson
// Copyright (C) 2012 Ryan Curtin
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_speye
//! @{



//! Generate a sparse matrix with the values along the main diagonal set to one
template<typename obj_type>
inline
obj_type
speye(const uword n_rows, const uword n_cols, const typename arma_SpMat_SpCol_SpRow_only<obj_type>::result* junk = NULL)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  if(is_SpCol<obj_type>::value == true)
    {
    arma_debug_check( (n_cols != 1), "speye(): incompatible size" );
    }
  else
  if(is_SpRow<obj_type>::value == true)
    {
    arma_debug_check( (n_rows != 1), "speye(): incompatible size" );
    }
  
  obj_type out;
  
  out.eye(n_rows, n_cols);
  
  return out;
  }



// Convenience shortcut method (no template parameter necessary)
inline
sp_mat
speye(const uword n_rows, const uword n_cols)
  {
  arma_extra_debug_sigprint();
  
  sp_mat out;
  
  out.eye(n_rows, n_cols);
  
  return out;
  }



//! @}
