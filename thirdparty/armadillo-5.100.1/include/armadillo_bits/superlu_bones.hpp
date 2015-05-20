// Copyright (C) 2015 Ryan Curtin
// Copyright (C) 2015 Conrad Sanderson
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

#if defined(ARMA_USE_SUPERLU)

extern "C"
  {
  extern void arma_wrapper(sgssv)(superlu::superlu_options_t*, superlu::SuperMatrix*, int*, int*, superlu::SuperMatrix*, superlu::SuperMatrix*, superlu::SuperMatrix*, superlu::SuperLUStat_t*, int*);
  extern void arma_wrapper(dgssv)(superlu::superlu_options_t*, superlu::SuperMatrix*, int*, int*, superlu::SuperMatrix*, superlu::SuperMatrix*, superlu::SuperMatrix*, superlu::SuperLUStat_t*, int*);
  extern void arma_wrapper(cgssv)(superlu::superlu_options_t*, superlu::SuperMatrix*, int*, int*, superlu::SuperMatrix*, superlu::SuperMatrix*, superlu::SuperMatrix*, superlu::SuperLUStat_t*, int*);
  extern void arma_wrapper(zgssv)(superlu::superlu_options_t*, superlu::SuperMatrix*, int*, int*, superlu::SuperMatrix*, superlu::SuperMatrix*, superlu::SuperMatrix*, superlu::SuperLUStat_t*, int*);
  
  extern void arma_wrapper(StatInit)(superlu::SuperLUStat_t*);
  extern void arma_wrapper(StatFree)(superlu::SuperLUStat_t*);
  extern void arma_wrapper(set_default_options)(superlu::superlu_options_t*);
  
  extern void arma_wrapper(Destroy_SuperNode_Matrix)(superlu::SuperMatrix*);
  extern void arma_wrapper(Destroy_CompCol_Matrix)(superlu::SuperMatrix*);
  extern void arma_wrapper(Destroy_SuperMatrix_Store)(superlu::SuperMatrix*);
  
  // We also need superlu_malloc() and superlu_free().
  // When using the original SuperLU code directly, you (the user) may
  // define USER_MALLOC and USER_FREE, but the joke is on you because
  // if you are linking against SuperLU and not compiling from scratch,
  // it won't actually make a difference anyway!  If you've compiled
  // SuperLU against a custom USER_MALLOC and USER_FREE, you're probably up
  // shit creek about a thousand different ways before you even get to this
  // code, so, don't do that!
  
  extern void* arma_wrapper(superlu_malloc)(size_t);
  extern void  arma_wrapper(superlu_free)(void*);
  }

#endif
