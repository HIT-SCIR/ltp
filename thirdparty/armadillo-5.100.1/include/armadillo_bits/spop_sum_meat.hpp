// Copyright (C) 2012 Ryan Curtin
// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup spop_sum
//! @{



template<typename T1>
arma_hot
inline
void
spop_sum::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1,spop_sum>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword dim = in.aux_uword_a;
  arma_debug_check((dim > 1), "sum(): incorrect usage. dim must be 0 or 1");
  
  const SpProxy<T1> p(in.m);
  
  if(p.is_alias(out) == false)
    {
    spop_sum::apply_noalias(out, p, dim);
    }
  else
    {
    SpMat<eT> tmp;
    
    spop_sum::apply_noalias(tmp, p, dim);
    
    out.steal_mem(tmp);
    }
  }



template<typename T1>
arma_hot
inline
void
spop_sum::apply_noalias(SpMat<typename T1::elem_type>& out, const SpProxy<T1>& p, const uword dim)
  {
  arma_extra_debug_sigprint();
  
  if(dim == 0) // find the sum in each column
    {
    out.zeros(1, p.get_n_cols());
    
    typename SpProxy<T1>::const_iterator_type it     = p.begin();
    typename SpProxy<T1>::const_iterator_type it_end = p.end();
    
    while(it != it_end)
      {
      out.at(0, it.col()) += (*it);
      ++it;
      }
    }
  else // find the sum in each row
    {
    out.zeros(p.get_n_rows(), 1);
    
    typename SpProxy<T1>::const_iterator_type it     = p.begin();
    typename SpProxy<T1>::const_iterator_type it_end = p.end();
    
    while(it != it_end)
      {
      out.at(it.row(), 0) += (*it);
      ++it;
      }
    }
  }



//! @}
