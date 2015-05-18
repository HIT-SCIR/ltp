// Copyright (C) 2014 Conrad Sanderson
// Copyright (C) 2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


namespace gmm_priv
{


template<typename eT>
inline
running_mean_scalar<eT>::running_mean_scalar()
  : counter(uword(0))
  , r_mean (   eT(0))
  {
  arma_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
running_mean_scalar<eT>::running_mean_scalar(const running_mean_scalar<eT>& in)
  : counter(in.counter)
  , r_mean (in.r_mean )
  {
  arma_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
const running_mean_scalar<eT>&
running_mean_scalar<eT>::operator=(const running_mean_scalar<eT>& in)
  {
  arma_extra_debug_sigprint();
  
  counter = in.counter;
  r_mean  = in.r_mean;
  
  return *this;
  }



template<typename eT>
arma_hot
inline
void
running_mean_scalar<eT>::operator() (const eT X)
  {
  arma_extra_debug_sigprint();
  
  counter++;
  
  if(counter > 1)
    {
    const eT old_r_mean = r_mean;
    
    r_mean = old_r_mean + (X - old_r_mean)/counter;
    }
  else
    {
    r_mean = X;
    }
  }



template<typename eT>
inline
void
running_mean_scalar<eT>::reset()
  {
  arma_extra_debug_sigprint();
  
  counter = 0;
  r_mean  = eT(0);
  }



template<typename eT>
inline
uword
running_mean_scalar<eT>::count() const
  {
  return counter;
  }



template<typename eT>
inline
eT
running_mean_scalar<eT>::mean() const
  {
  return r_mean;
  }



//
//
//



template<typename eT>
inline
running_mean_vec<eT>::running_mean_vec()
  : last_i (0)
  , counter(0)
  {
  arma_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
running_mean_vec<eT>::running_mean_vec(const running_mean_vec<eT>& in)
  : last_i (in.last_i )
  , counter(in.counter)
  , r_mean (in.r_mean )
  {
  arma_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
const running_mean_vec<eT>&
running_mean_vec<eT>::operator=(const running_mean_vec<eT>& in)
  {
  arma_extra_debug_sigprint();
  
  last_i  = in.last_i; 
  counter = in.counter;
  r_mean  = in.r_mean;
  
  return *this;
  }



template<typename eT>
arma_hot
inline
void
running_mean_vec<eT>::operator() (const Col<eT>& X, const uword index)
  {
  arma_extra_debug_sigprint();
  
  last_i = index;
  
  counter++;
  
  if(counter > 1)
    {
    const uword n_elem      = r_mean.n_elem;
    
          eT*   r_mean_mem  = r_mean.memptr();
    const eT*   X_mem       = X.memptr();
    
    for(uword i=0; i<n_elem; ++i)
      {
      const eT r_mean_val = r_mean_mem[i];
      
      r_mean_mem[i] = r_mean_val + (X_mem[i] - r_mean_val)/counter;
      }
    }
  else
    {
    r_mean = X;
    }
  }



template<typename eT>
inline
void
running_mean_vec<eT>::reset()
  {
  arma_extra_debug_sigprint();
  
  last_i  = 0;
  counter = 0;
  
  r_mean.reset();
  }



template<typename eT>
inline
uword
running_mean_vec<eT>::last_index() const
  {
  return last_i;
  }



template<typename eT>
inline
uword
running_mean_vec<eT>::count() const
  {
  return counter;
  }



template<typename eT>
inline
const Col<eT>&
running_mean_vec<eT>::mean() const
  {
  return r_mean;
  }




//
//
//




template<typename eT>
arma_inline
arma_hot
eT
distance<eT, uword(1)>::eval(const uword N, const eT* A, const eT* B, const eT*)
  {
  eT acc1 = eT(0);
  eT acc2 = eT(0);
  
  uword i,j;
  for(i=0, j=1; j<N; i+=2, j+=2)
    {
    eT tmp_i = A[i];
    eT tmp_j = A[j];
    
    tmp_i -= B[i];
    tmp_j -= B[j];
    
    acc1 += tmp_i*tmp_i;
    acc2 += tmp_j*tmp_j;
    }
  
  if(i < N)
    {
    const eT tmp_i = A[i] - B[i];
    
    acc1 += tmp_i*tmp_i;
    }
  
  return (acc1 + acc2);
  }



template<typename eT>
arma_inline
arma_hot
eT
distance<eT, uword(2)>::eval(const uword N, const eT* A, const eT* B, const eT* C)
  {
  eT acc1 = eT(0);
  eT acc2 = eT(0);
  
  uword i,j;
  for(i=0, j=1; j<N; i+=2, j+=2)
    {
    eT tmp_i = A[i];
    eT tmp_j = A[j];
    
    tmp_i -= B[i];
    tmp_j -= B[j];
    
    acc1 += (tmp_i*tmp_i) * C[i];
    acc2 += (tmp_j*tmp_j) * C[j];
    }
  
  if(i < N)
    {
    const eT tmp_i = A[i] - B[i];
    
    acc1 += (tmp_i*tmp_i) * C[i];
    }
  
  return (acc1 + acc2);
  }

}
