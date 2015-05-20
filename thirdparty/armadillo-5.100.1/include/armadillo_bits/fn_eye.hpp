// Copyright (C) 2008-2012 Conrad Sanderson
// Copyright (C) 2008-2012 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_eye
//! @{



arma_inline
const Gen<mat, gen_ones_diag>
eye(const uword n_rows, const uword n_cols)
  {
  arma_extra_debug_sigprint();
  
  return Gen<mat, gen_ones_diag>(n_rows, n_cols);
  }



template<typename obj_type>
arma_inline
const Gen<obj_type, gen_ones_diag>
eye(const uword n_rows, const uword n_cols, const typename arma_Mat_Col_Row_only<obj_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  if(is_Col<obj_type>::value == true)
    {
    arma_debug_check( (n_cols != 1), "eye(): incompatible size" );
    }
  else
  if(is_Row<obj_type>::value == true)
    {
    arma_debug_check( (n_rows != 1), "eye(): incompatible size" );
    }
  
  return Gen<obj_type, gen_ones_diag>(n_rows, n_cols);
  }



//! @}
