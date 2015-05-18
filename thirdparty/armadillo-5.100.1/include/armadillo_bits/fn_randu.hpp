// Copyright (C) 2008-2013 Conrad Sanderson
// Copyright (C) 2008-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_randu
//! @{


inline
double
randu()
  {
  return arma_rng::randu<double>();
  }


template<typename eT>
inline
typename arma_scalar_only<eT>::result
randu()
  {
  return eT(arma_rng::randu<eT>());
  }



//! Generate a vector with all elements set to random values in the [0,1] interval (uniform distribution)
arma_inline
const Gen<vec, gen_randu>
randu(const uword n_elem)
  {
  arma_extra_debug_sigprint();
  
  return Gen<vec, gen_randu>(n_elem, 1);
  }



template<typename obj_type>
arma_inline
const Gen<obj_type, gen_randu>
randu(const uword n_elem, const arma_empty_class junk1 = arma_empty_class(), const typename arma_Mat_Col_Row_only<obj_type>::result* junk2 = 0)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  if(is_Row<obj_type>::value == true)
    {
    return Gen<obj_type, gen_randu>(1, n_elem);
    }
  else
    {
    return Gen<obj_type, gen_randu>(n_elem, 1);
    }
  }



//! Generate a dense matrix with all elements set to random values in the [0,1] interval (uniform distribution)
arma_inline
const Gen<mat, gen_randu>
randu(const uword n_rows, const uword n_cols)
  {
  arma_extra_debug_sigprint();
  
  return Gen<mat, gen_randu>(n_rows, n_cols);
  }



template<typename obj_type>
arma_inline
const Gen<obj_type, gen_randu>
randu(const uword n_rows, const uword n_cols, const typename arma_Mat_Col_Row_only<obj_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  if(is_Col<obj_type>::value == true)
    {
    arma_debug_check( (n_cols != 1), "randu(): incompatible size" );
    }
  else
  if(is_Row<obj_type>::value == true)
    {
    arma_debug_check( (n_rows != 1), "randu(): incompatible size" );
    }
  
  return Gen<obj_type, gen_randu>(n_rows, n_cols);
  }



arma_inline
const GenCube<cube::elem_type, gen_randu>
randu(const uword n_rows, const uword n_cols, const uword n_slices)
  {
  arma_extra_debug_sigprint();
  
  return GenCube<cube::elem_type, gen_randu>(n_rows, n_cols, n_slices);
  }



template<typename cube_type>
arma_inline
const GenCube<typename cube_type::elem_type, gen_randu>
randu(const uword n_rows, const uword n_cols, const uword n_slices, const typename arma_Cube_only<cube_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return GenCube<typename cube_type::elem_type, gen_randu>(n_rows, n_cols, n_slices);
  }



//! @}
