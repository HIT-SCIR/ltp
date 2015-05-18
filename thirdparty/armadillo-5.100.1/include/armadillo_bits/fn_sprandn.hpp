// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_sprandn
//! @{



//! Generate a sparse matrix with a randomly selected subset of the elements
//! set to random values from a Gaussian distribution with zero mean and unit variance
template<typename obj_type>
inline
obj_type
sprandn
  (
  const uword  n_rows,
  const uword  n_cols,
  const double density,
  const typename arma_SpMat_SpCol_SpRow_only<obj_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  if(is_SpCol<obj_type>::value == true)
    {
    arma_debug_check( (n_cols != 1), "sprandn(): incompatible size" );
    }
  else
  if(is_SpRow<obj_type>::value == true)
    {
    arma_debug_check( (n_rows != 1), "sprandn(): incompatible size" );
    }
  
  obj_type out;
  
  out.sprandn(n_rows, n_cols, density);
  
  return out;
  }



inline
sp_mat
sprandn(const uword n_rows, const uword n_cols, const double density)
  {
  arma_extra_debug_sigprint();
  
  sp_mat out;
  
  out.sprandn(n_rows, n_cols, density);
  
  return out;
  }



//! Generate a sparse matrix with the non-zero values in the same locations as in the given sparse matrix X,
//! with the non-zero values set to random values from a Gaussian distribution with zero mean and unit variance
template<typename T1>
inline
SpMat<typename T1::elem_type>
sprandn(const SpBase<typename T1::elem_type, T1>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  SpMat<eT> out( X.get_ref() );
  
  arma_rng::randn<eT>::fill( access::rwp(out.values), out.n_nonzero );
  
  return out;
  }



//! @}
