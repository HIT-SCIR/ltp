// Copyright (C) 2014 Conrad Sanderson
// Copyright (C) 2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



template<typename T1>
inline
typename
enable_if2
  <
  is_real<typename T1::pod_type>::value,
  const Op<T1,op_expmat>
  >::result
expmat(const Base<typename T1::elem_type,T1>& A)
  {
  arma_extra_debug_sigprint();
  
  return Op<T1,op_expmat>( A.get_ref() );
  }
