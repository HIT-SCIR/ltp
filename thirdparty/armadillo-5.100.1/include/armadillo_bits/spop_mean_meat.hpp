// Copyright (C) 2012 Ryan Curtin
// Copyright (C) 2012 Conrad Sanderson
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup spop_mean
//! @{



template<typename T1>
inline
void
spop_mean::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1, spop_mean>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword dim = in.aux_uword_a;
  arma_debug_check((dim > 1), "mean(): incorrect usage. dim must be 0 or 1");
  
  SpProxy<T1> p(in.m);
  
  if(p.is_alias(out) == false)
    {
    spop_mean::apply_noalias(out, p, dim);
    }
  else
    {
    SpMat<eT> tmp;
    
    spop_mean::apply_noalias(tmp, p, dim);
    
    out.steal_mem(tmp);
    }
  }



template<typename T1>
inline
void
spop_mean::apply_noalias
  (
        SpMat<typename T1::elem_type>& out_ref,
  const SpProxy<T1>&                   p,
  const uword                          dim
  )
  {
  arma_extra_debug_sigprint();

  typedef typename T1::elem_type eT;
  
  const uword p_n_rows = p.get_n_rows();
  const uword p_n_cols = p.get_n_cols();

  if (dim == 0)
    {
    arma_extra_debug_print("spop_mean::apply_noalias(), dim = 0");

    out_ref.set_size((p_n_rows > 0) ? 1 : 0, p_n_cols);

    if(p_n_rows > 0)
      {
      for(uword col = 0; col < p_n_cols; ++col)
        {
        // Do we have to use an iterator or can we use memory directly?
        if(SpProxy<T1>::must_use_iterator == true)
          {
          typename SpProxy<T1>::const_iterator_type it  = p.begin_col(col);
          typename SpProxy<T1>::const_iterator_type end = p.begin_col(col + 1);
          
          const uword n_zero = p.get_n_rows() - (end.pos() - it.pos());
          
          out_ref.at(col) = spop_mean::iterator_mean(it, end, n_zero, eT(0));
          }
        else
          {
          out_ref.at(col) = spop_mean::direct_mean
            (
            &p.get_values()[p.get_col_ptrs()[col]],
            p.get_col_ptrs()[col + 1] - p.get_col_ptrs()[col],
            p.get_n_rows()
            );
          }
        }
      }
    }
  else if (dim == 1)
    {
    arma_extra_debug_print("spop_mean::apply_noalias(), dim = 1");
    
    out_ref.set_size(p_n_rows, (p_n_cols > 0) ? 1 : 0);
    
    if(p_n_cols > 0)
      {
      for(uword row = 0; row < p_n_rows; ++row)
        {
        // We must use an iterator regardless of how it is stored.
        typename SpProxy<T1>::const_row_iterator_type it  = p.begin_row(row);
        typename SpProxy<T1>::const_row_iterator_type end = p.end_row(row);
        
        const uword n_zero = p.get_n_cols() - (end.pos() - it.pos());
        
        out_ref.at(row) = spop_mean::iterator_mean(it, end, n_zero, eT(0));
        }
      }
    }
  }



template<typename eT>
inline
eT
spop_mean::direct_mean
  (
  const eT* const X,
  const uword length,
  const uword N
  )
  {
  arma_extra_debug_sigprint();

  typedef typename get_pod_type<eT>::result T;

  const eT result = arrayops::accumulate(X, length) / T(N);

  return arma_isfinite(result) ? result : spop_mean::direct_mean_robust(X, length, N);
  }



template<typename eT>
inline
eT
spop_mean::direct_mean_robust
  (
  const eT* const X,
  const uword length,
  const uword N
  )
  {
  arma_extra_debug_sigprint();

  typedef typename get_pod_type<eT>::result T;

  uword i, j;

  eT r_mean = eT(0);

  const uword diff = (N - length); // number of zeros

  for(i = 0, j = 1; j < length; i += 2, j += 2)
    {
    const eT Xi = X[i];
    const eT Xj = X[j];

    r_mean += (Xi - r_mean) / T(diff + j);
    r_mean += (Xj - r_mean) / T(diff + j + 1);
    }

  if(i < length)
    {
    const eT Xi = X[i];

    r_mean += (Xi - r_mean) / T(diff + i + 1);
    }

  return r_mean;
  }



template<typename T1>
inline
typename T1::elem_type
spop_mean::mean_all(const SpBase<typename T1::elem_type, T1>& X)
  {
  arma_extra_debug_sigprint();

  SpProxy<T1> p(X.get_ref());

  if (SpProxy<T1>::must_use_iterator == true)
    {
    typename SpProxy<T1>::const_iterator_type it  = p.begin();
    typename SpProxy<T1>::const_iterator_type end = p.end();

    return spop_mean::iterator_mean(it, end, p.get_n_elem() - p.get_n_nonzero(), typename T1::elem_type(0));
    }
  else // must_use_iterator == false; that is, we can directly access the values array
    {
    return spop_mean::direct_mean(p.get_values(), p.get_n_nonzero(), p.get_n_elem());
    }
  }



template<typename T1, typename eT>
inline
eT
spop_mean::iterator_mean(T1& it, const T1& end, const uword n_zero, const eT junk)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);

  typedef typename get_pod_type<eT>::result T;

  eT sum = eT(0);

  T1 backup_it(it); // in case we have to use robust iterator_mean

  const uword it_begin_pos = it.pos();

  while (it != end)
    {
    sum += (*it);
    ++it;
    }

  const eT result = sum / T(n_zero + (it.pos() - it_begin_pos));

  return arma_isfinite(result) ? result : spop_mean::iterator_mean_robust(backup_it, end, n_zero, eT(0));
  }



template<typename T1, typename eT>
inline
eT
spop_mean::iterator_mean_robust(T1& it, const T1& end, const uword n_zero, const eT junk)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);

  typedef typename get_pod_type<eT>::result T;

  eT r_mean = eT(0);

  const uword it_begin_pos = it.pos();

  while (it != end)
    {
    r_mean += ((*it - r_mean) / T(n_zero + (it.pos() - it_begin_pos) + 1));
    ++it;
    }

  return r_mean;
  }



//! @}
