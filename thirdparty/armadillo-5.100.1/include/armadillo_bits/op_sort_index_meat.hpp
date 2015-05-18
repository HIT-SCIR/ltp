// Copyright (C) 2009-2015 Conrad Sanderson
// Copyright (C) 2009-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_sort_index
//! @{



template<typename T1, bool sort_stable>
inline
bool
arma_sort_index_helper(Mat<uword>& out, const Proxy<T1>& P, const uword sort_type, typename arma_not_cx<typename T1::elem_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  
  const uword n_elem = P.get_n_elem();
  
  out.set_size(n_elem, 1);
  
  std::vector< arma_sort_index_packet<eT, uword> > packet_vec(n_elem);
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    for(uword i=0; i<n_elem; ++i)
      {
      const eT val = P[i];
      
      if(is_finite(val) == false)  { out.reset(); return false; }
      
      packet_vec[i].val   = val;
      packet_vec[i].index = i;
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    uword i = 0;
    
    for(uword col=0; col < n_cols; ++col)
    for(uword row=0; row < n_rows; ++row)
      {
      const eT val = P.at(row,col);
      
      if(is_finite(val) == false)  { out.reset(); return false; }
      
      packet_vec[i].val   = val;
      packet_vec[i].index = i;
      
      ++i;
      }
    }
  
  
  if(sort_type == 0)
    {
    // ascend
    
    arma_sort_index_helper_ascend comparator;
    
    if(sort_stable == false)
      {
      std::sort( packet_vec.begin(), packet_vec.end(), comparator );
      }
    else
      {
      std::stable_sort( packet_vec.begin(), packet_vec.end(), comparator );
      }
    }
  else
    {
    // descend
    
    arma_sort_index_helper_descend comparator;
    
    if(sort_stable == false)
      {
      std::sort( packet_vec.begin(), packet_vec.end(), comparator );
      }
    else
      {
      std::stable_sort( packet_vec.begin(), packet_vec.end(), comparator );
      }
    }
  
  uword* out_mem = out.memptr();
  
  for(uword i=0; i<n_elem; ++i)
    {
    out_mem[i] = packet_vec[i].index;
    }
  
  return true;
  }



template<typename T1, bool sort_stable>
inline
bool
arma_sort_index_helper(Mat<uword>& out, const Proxy<T1>& P, const uword sort_type, typename arma_cx_only<typename T1::elem_type>::result* junk = 0)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type            eT;
  typedef typename get_pod_type<eT>::result  T;
  
  const uword n_elem = P.get_n_elem();
  
  out.set_size(n_elem, 1);
  
  std::vector< arma_sort_index_packet<T, uword> > packet_vec(n_elem);
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    for(uword i=0; i<n_elem; ++i)
      {
      const T val = std::abs(P[i]);
      
      if(is_finite(val) == false)  { out.reset(); return false; }
      
      packet_vec[i].val   = val;
      packet_vec[i].index = i;
      }
    }
  else
    {
    const uword n_rows = P.get_n_rows();
    const uword n_cols = P.get_n_cols();
    
    uword i = 0;
    
    for(uword col=0; col < n_cols; ++col)
    for(uword row=0; row < n_rows; ++row)
      {
      const T val = std::abs(P.at(row,col));
      
      if(is_finite(val) == false)  { out.reset(); return false; }
      
      packet_vec[i].val   = val;
      packet_vec[i].index = i;
      
      ++i;
      }
    }
  
  
  if(sort_type == 0)
    {
    // ascend
    
    arma_sort_index_helper_ascend comparator;
    
    if(sort_stable == false)
      {
      std::sort( packet_vec.begin(), packet_vec.end(), comparator );
      }
    else
      {
      std::stable_sort( packet_vec.begin(), packet_vec.end(), comparator );
      }
    }
  else
    {
    // descend
    
    arma_sort_index_helper_descend comparator;
    
    if(sort_stable == false)
      {
      std::sort( packet_vec.begin(), packet_vec.end(), comparator );
      }
    else
      {
      std::stable_sort( packet_vec.begin(), packet_vec.end(), comparator );
      }
    }
  
  uword* out_mem = out.memptr();
  
  for(uword i=0; i<n_elem; ++i)
    {
    out_mem[i] = packet_vec[i].index;
    }
  
  return true;
  }



template<typename T1>
inline
bool
op_sort_index::apply_noalias(Mat<uword>& out, const Proxy<T1>& P, const uword sort_type)
  {
  arma_extra_debug_sigprint();
  
  return arma_sort_index_helper<T1,false>(out, P, sort_type);
  }



template<typename T1>
inline
void
op_sort_index::apply(Mat<uword>& out, const mtOp<uword,T1,op_sort_index>& in)
  {
  arma_extra_debug_sigprint();
  
  const Proxy<T1> P(in.m);
  
  if(P.get_n_elem() == 0)  { out.set_size(0,1); return; }
  
  const uword sort_type = in.aux_uword_a;
  
  bool all_finite = false;
  
  if(P.is_alias(out))
    {
    Mat<uword> out2;
    
    all_finite = op_sort_index::apply_noalias(out2, P, sort_type);
    
    out.steal_mem(out2);
    }
  else
    {
    all_finite = op_sort_index::apply_noalias(out, P, sort_type);
    }
  
  arma_debug_check( (all_finite == false), "sort_index(): detected non-finite values" );
  }



template<typename T1>
inline
bool
op_stable_sort_index::apply_noalias(Mat<uword>& out, const Proxy<T1>& P, const uword sort_type)
  {
  arma_extra_debug_sigprint();
  
  return arma_sort_index_helper<T1,true>(out, P, sort_type);
  }



template<typename T1>
inline
void
op_stable_sort_index::apply(Mat<uword>& out, const mtOp<uword,T1,op_stable_sort_index>& in)
  {
  arma_extra_debug_sigprint();
  
  const Proxy<T1> P(in.m);
  
  if(P.get_n_elem() == 0)  { out.set_size(0,1); return; }
  
  const uword sort_type = in.aux_uword_a;
  
  bool all_finite = false;
  
  if(P.is_alias(out))
    {
    Mat<uword> out2;
    
    all_finite = op_stable_sort_index::apply_noalias(out2, P, sort_type);
    
    out.steal_mem(out2);
    }
  else
    {
    all_finite = op_stable_sort_index::apply_noalias(out, P, sort_type);
    }
  
  arma_debug_check( (all_finite == false), "stable_sort_index(): detected non-finite values" );
  }



//! @}
