// Copyright (C) 2012 Conrad Sanderson
// Copyright (C) 2012 NICTA (www.nicta.com.au)
// Copyright (C) 2012 Boris Sabanin
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



template<typename T1, typename T2>
inline
const mtGlue<uword,T1,T2,glue_histc>
histc
  (
  const Base<typename T1::elem_type,T1>& A,
  const Base<typename T1::elem_type,T2>& B,
  const uword dim = 0,
  const typename arma_not_cx<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return mtGlue<uword,T1,T2,glue_histc>( A.get_ref(), B.get_ref(), dim );
  }
