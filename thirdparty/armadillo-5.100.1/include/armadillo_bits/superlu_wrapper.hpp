// Copyright (C) 2015 Ryan Curtin
// Copyright (C) 2015 Conrad Sanderson
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



#if defined(ARMA_USE_SUPERLU)

//! \namespace superlu namespace for SuperLU functions
namespace superlu
  {
  
  template<typename eT>
  inline
  void
  gssv(superlu_options_t* options, SuperMatrix* A, int* perm_c, int* perm_r, SuperMatrix* L, SuperMatrix* U, SuperMatrix* B, SuperLUStat_t* stat, int* info)
    {
    arma_type_check(( is_supported_blas_type<eT>::value == false ));
    
    if(is_float<eT>::value)
      {
      arma_wrapper(sgssv)(options, A, perm_c, perm_r, L, U, B, stat, info);
      }
    else
    if(is_double<eT>::value)
      {
      arma_wrapper(dgssv)(options, A, perm_c, perm_r, L, U, B, stat, info);
      }
    else
    if(is_supported_complex_float<eT>::value)
      {
      arma_wrapper(cgssv)(options, A, perm_c, perm_r, L, U, B, stat, info);
      }
    else
    if(is_supported_complex_double<eT>::value)
      {
      arma_wrapper(zgssv)(options, A, perm_c, perm_r, L, U, B, stat, info);
      }
    }
  
  
  
  inline
  void
  init_stat(SuperLUStat_t* stat)
    {
    arma_wrapper(StatInit)(stat);
    }


  inline
  void
  free_stat(SuperLUStat_t* stat)
    {
    arma_wrapper(StatFree)(stat);
    }
  
  
  
  inline
  void
  set_default_opts(superlu_options_t* opts)
    {
    arma_wrapper(set_default_options)(opts);
    }
  
  
  
  inline
  void
  destroy_supernode_mat(SuperMatrix* a)
    {
    arma_wrapper(Destroy_SuperNode_Matrix)(a);
    }



  inline
  void
  destroy_compcol_mat(SuperMatrix* a)
    {
    arma_wrapper(Destroy_CompCol_Matrix)(a);
    }



  inline
  void
  destroy_dense_mat(SuperMatrix* a)
    {
    arma_wrapper(Destroy_SuperMatrix_Store)(a);
    }
  
  
  
  inline
  void*
  malloc(size_t N)
    {
    return arma_wrapper(superlu_malloc)(N);
    }
  
  
  
  inline
  void
  free(void* mem)
    {
    arma_wrapper(superlu_free)(mem);
    }
  
  } // namespace superlu

#endif
