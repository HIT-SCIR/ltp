// Copyright (C) 2012-2014 Ryan Curtin
// Copyright (C) 2012-2014 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup spop_min
//! @{



template<typename T1>
inline
void
spop_min::apply(SpMat<typename T1::elem_type>& out, const SpOp<T1,spop_min>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword dim = in.aux_uword_a;
  arma_debug_check((dim > 1), "min(): incorrect usage. dim must be 0 or 1");
  
  const SpProxy<T1> p(in.m);
  
  if(p.is_alias(out) == false)
    {
    spop_min::apply_noalias(out, p, dim);
    }
  else
    {
    SpMat<eT> tmp;
    
    spop_min::apply_noalias(tmp, p, dim);
    
    out.steal_mem(tmp);
    }
  }



template<typename T1>
inline
void
spop_min::apply_noalias
  (
        SpMat<typename T1::elem_type>& result,
  const SpProxy<T1>&                   p,
  const uword                          dim,
  const typename arma_not_cx<typename T1::elem_type>::result* junk
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;

  if(dim == 0)
    {
    // minimum in each column
    result.set_size(1, p.get_n_cols());
    
    if(p.get_n_nonzero() == 0)  { return; }
    
    typename SpProxy<T1>::const_iterator_type it     = p.begin();
    typename SpProxy<T1>::const_iterator_type it_end = p.end();
    
    uword cur_col     = it.col();
    uword elem_in_col = 1;
    
    eT cur_min = (*it);
    ++it;
    
    while(it != it_end)
      {
      if(it.col() != cur_col)
        {
        // was the column full?
        if(elem_in_col == p.get_n_rows())
          {
          result.at(0, cur_col) = cur_min;
          }
        else
          {
          result.at(0, cur_col) = std::min(eT(0), cur_min);
          }

        cur_col     = it.col();
        elem_in_col = 0;
        
        cur_min = (*it);
        }
      else
        {
        cur_min = std::min(cur_min, *it);
        }

      ++elem_in_col;
      ++it;
      }

    if(elem_in_col == p.get_n_rows())
      {
      result.at(0, cur_col) = cur_min;
      }
    else
      {
      result.at(0, cur_col) = std::min(eT(0), cur_min);
      }
    }
  else
    {
    // minimum in each row
    result.set_size(p.get_n_rows(), 1);
    
    if(p.get_n_nonzero() == 0)  { return; }
    
    typename SpProxy<T1>::const_row_iterator_type it = p.begin_row();
    
    uword cur_row     = it.row();
    uword elem_in_row = 1;
    
    eT cur_min = (*it);
    ++it;
    
    while(it.pos() < p.get_n_nonzero())
      {
      if(it.row() != cur_row)
        {
        // was the row full?
        if(elem_in_row == p.get_n_cols())
          {
          result.at(cur_row, 0) = cur_min;
          }
        else
          {
          result.at(cur_row, 0) = std::min(eT(0), cur_min);
          }

        cur_row     = it.row();
        elem_in_row = 0;
        
        cur_min = (*it);
        }
      else
        {
        cur_min = std::min(cur_min, *it);
        }

      ++elem_in_row;
      ++it;
      }

    if(elem_in_row == p.get_n_cols())
      {
      result.at(cur_row, 0) = cur_min;
      }
    else
      {
      result.at(cur_row, 0) = std::min(eT(0), cur_min);
      }
    }
  }



template<typename T1>
inline
typename T1::elem_type
spop_min::vector_min
  (
  const T1& x,
  const typename arma_not_cx<typename T1::elem_type>::result* junk
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  
  const SpProxy<T1> p(x);
  
  if(p.get_n_nonzero() == 0)  { return eT(0); }
  
  if(SpProxy<T1>::must_use_iterator == false)
    {
    // direct access of values
    if(p.get_n_nonzero() == p.get_n_elem())
      {
      return op_min::direct_min(p.get_values(), p.get_n_nonzero());
      }
    else
      {
      return std::min(eT(0), op_min::direct_min(p.get_values(), p.get_n_nonzero()));
      }
    }
  else
    {
    // use iterator
    typename SpProxy<T1>::const_iterator_type it     = p.begin();
    typename SpProxy<T1>::const_iterator_type it_end = p.end();
    
    eT result = (*it);
    ++it;
    
    while(it != it_end)
      {
      if((*it) < result)  { result = (*it); }
      
      ++it;
      }
    
    if(p.get_n_nonzero() == p.get_n_elem())
      {
      return result;
      }
    else
      {
      return std::min(eT(0), result);
      }
    }
  }



template<typename T1>
inline
typename arma_not_cx<typename T1::elem_type>::result
spop_min::min(const SpBase<typename T1::elem_type, T1>& X)
  {
  arma_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const SpProxy<T1> P(X.get_ref());

  const uword n_elem    = P.get_n_elem();
  const uword n_nonzero = P.get_n_nonzero();

  arma_debug_check( (n_elem == 0), "min(): given object has no elements");

  eT min_val = priv::most_pos<eT>();

  if(SpProxy<T1>::must_use_iterator == true)
    {
    // We have to iterate over the elements.
    typedef typename SpProxy<T1>::const_iterator_type it_type;

    it_type it     = P.begin();
    it_type it_end = P.end();

    while (it != it_end)
      {
      if ((*it) < min_val)  { min_val = *it; }
      
      ++it;
      }
    }
  else
    {
    // We can do direct access of the values, row_indices, and col_ptrs.
    // We don't need the location of the min value, so we can just call out to
    // other functions...
    min_val = op_min::direct_min(P.get_values(), n_nonzero);
    }

  if(n_elem == n_nonzero)
    {
    return min_val;
    }
  else
    {
    return std::min(eT(0), min_val);
    }
  }



template<typename T1>
inline
typename arma_not_cx<typename T1::elem_type>::result
spop_min::min_with_index(const SpProxy<T1>& P, uword& index_of_min_val)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_elem    = P.get_n_elem();
  const uword n_nonzero = P.get_n_nonzero();
  const uword n_rows    = P.get_n_rows();
  
  arma_debug_check( (n_elem == 0), "min(): given object has no elements");
  
  eT min_val = priv::most_pos<eT>();
  
  if(SpProxy<T1>::must_use_iterator == true)
    {
    // We have to iterate over the elements.
    typedef typename SpProxy<T1>::const_iterator_type it_type;
    
    it_type it     = P.begin();
    it_type it_end = P.end();
    
    while (it != it_end)
      {
      if ((*it) < min_val)
        {
        min_val = *it;
        index_of_min_val = it.row() + it.col() * n_rows;
        }
      
      ++it;
      }
    }
  else
    {
    // We can do direct access.
    min_val = op_min::direct_min(P.get_values(), n_nonzero, index_of_min_val);
    
    // Convert to actual position in matrix.
    const uword row = P.get_row_indices()[index_of_min_val];
    uword col = 0;
    while (P.get_col_ptrs()[++col] < index_of_min_val + 1) { }
    index_of_min_val = (col - 1) * n_rows + row;
    }
  
  
  if(n_elem != n_nonzero)
    {
    min_val = std::min(eT(0), min_val);

    // If the min_val is a nonzero element, we need its actual position in the matrix.
    if(min_val == eT(0))
      {
      // Find first zero element.
      uword last_row = 0;
      uword last_col = 0;
      
      typedef typename SpProxy<T1>::const_iterator_type it_type;
      
      it_type it     = P.begin();
      it_type it_end = P.end();
      
      while (it != it_end)
        {
        // Have we moved more than one position from the last place?
        if ((it.col() == last_col) && (it.row() - last_row > 1))
          {
          index_of_min_val = it.col() * n_rows + last_row + 1;
          break;
          }
        else if ((it.col() >= last_col + 1) && (last_row < n_rows - 1))
          {
          index_of_min_val = last_col * n_rows + last_row + 1;
          break;
          }
        else if ((it.col() == last_col + 1) && (it.row() > 0))
          {
          index_of_min_val = it.col() * n_rows;
          break;
          }
        else if (it.col() > last_col + 1)
          {
          index_of_min_val = (last_col + 1) * n_rows;
          break;
          }
        
        last_row = it.row();
        last_col = it.col();
        ++it;
        }
      }
    }
  
  return min_val;
  }



template<typename T1>
inline
void
spop_min::apply_noalias
  (
        SpMat<typename T1::elem_type>& result,
  const SpProxy<T1>&                   p,
  const uword                          dim,
  const typename arma_cx_only<typename T1::elem_type>::result* junk
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type            eT;
  typedef typename get_pod_type<eT>::result  T;
  
  if(dim == 0)
    {
    // minimum in each column
    result.set_size(1, p.get_n_cols());
    
    if(p.get_n_nonzero() == 0)  { return; }
    
    typename SpProxy<T1>::const_iterator_type it     = p.begin();
    typename SpProxy<T1>::const_iterator_type it_end = p.end();
    
    uword cur_col     = it.col();
    uword elem_in_col = 1;
    
    eT cur_min_orig = *it;
     T cur_min_abs  = std::abs(cur_min_orig);
    
    ++it;
    
    while(it != it_end)
      {
      if(it.col() != cur_col)
        {
        // was the column full?
        if(elem_in_col == p.get_n_rows())
          {
          result.at(0, cur_col) = cur_min_orig;
          }
        else
          {
          eT val1 = eT(0);
          
          result.at(0, cur_col) = ( std::abs(val1) < cur_min_abs ) ? val1 : cur_min_orig;
          }
        
        cur_col     = it.col();
        elem_in_col = 0;
        
        cur_min_orig = *it;
        cur_min_abs  = std::abs(cur_min_orig);
        }
      else
        {
        eT val1_orig = *it;
         T val1_abs  = std::abs(val1_orig);
        
        if( val1_abs < cur_min_abs )
          {
          cur_min_abs  = val1_abs;
          cur_min_orig = val1_orig;
          }
        }

      ++elem_in_col;
      ++it;
      }

    if(elem_in_col == p.get_n_rows())
      {
      result.at(0, cur_col) = cur_min_orig;
      }
    else
      {
      eT val1 = eT(0);
      
      result.at(0, cur_col) = ( std::abs(val1) < cur_min_abs ) ? val1 : cur_min_orig;
      }
    }
  else
    {
    // minimum in each row
    result.set_size(p.get_n_rows(), 1);
    
    if(p.get_n_nonzero() == 0)  { return; }
    
    typename SpProxy<T1>::const_row_iterator_type it = p.begin_row();
    
    uword cur_row     = it.row();
    uword elem_in_row = 1;
    
    eT cur_min_orig = *it;
     T cur_min_abs  = std::abs(cur_min_orig);
    
    ++it;
    
    while(it.pos() < p.get_n_nonzero())
      {
      if(it.row() != cur_row)
        {
        // was the row full?
        if(elem_in_row == p.get_n_cols())
          {
          result.at(cur_row, 0) = cur_min_orig;
          }
        else
          {
          eT val1 = eT(0);
          
          result.at(cur_row, 0) = ( std::abs(val1) < cur_min_abs ) ? val1 : cur_min_orig;
          }

        cur_row = it.row();
        elem_in_row = 0;
        
        cur_min_orig = *it;
        cur_min_abs  = std::abs(cur_min_orig);
        }
      else
        {
        eT val1_orig = *it;
         T val1_abs  = std::abs(val1_orig);
        
        if( val1_abs < cur_min_abs )
          {
          cur_min_abs  = val1_abs;
          cur_min_orig = val1_orig;
          }
        }

      ++elem_in_row;
      ++it;
      }

    if(elem_in_row == p.get_n_cols())
      {
      result.at(cur_row, 0) = cur_min_orig;
      }
    else
      {
      eT val1 = eT(0);
      
      result.at(cur_row, 0) = ( std::abs(val1) < cur_min_abs ) ? val1 : cur_min_orig;
      }
    }
  }



template<typename T1>
inline
typename T1::elem_type
spop_min::vector_min
  (
  const T1& x,
  const typename arma_cx_only<typename T1::elem_type>::result* junk
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type            eT;
  typedef typename get_pod_type<eT>::result  T;

  const SpProxy<T1> p(x);

  if(p.get_n_nonzero() == 0)  { return eT(0); }
  
  if(SpProxy<T1>::must_use_iterator == false)
    {
    // direct access of values
    if(p.get_n_nonzero() == p.get_n_elem())
      {
      return op_min::direct_min(p.get_values(), p.get_n_nonzero());
      }
    else
      {
      const eT val1 = eT(0);
      const eT val2 = op_min::direct_min(p.get_values(), p.get_n_nonzero());
      
      return ( std::abs(val1) < std::abs(val2) ) ? val1 : val2;
      }
    }
  else
    {
    // use iterator
    typename SpProxy<T1>::const_iterator_type it     = p.begin();
    typename SpProxy<T1>::const_iterator_type it_end = p.end();

    eT best_val_orig = *it;
     T best_val_abs  = std::abs(best_val_orig);
    
    ++it;
    
    while(it != it_end)
      {
      eT val_orig = *it;
       T val_abs  = std::abs(val_orig);
      
      if(val_abs < best_val_abs)
        {
        best_val_abs  = val_abs;
        best_val_orig = val_orig;
        }

      ++it;
      }

    if(p.get_n_nonzero() == p.get_n_elem())
      {
      return best_val_orig;
      }
    else
      {
      const eT val1 = eT(0);
      
      return ( std::abs(val1) < best_val_abs ) ? val1 : best_val_orig;
      }
    }
  }



template<typename T1>
inline
typename arma_cx_only<typename T1::elem_type>::result
spop_min::min(const SpBase<typename T1::elem_type, T1>& X)
  {
  arma_extra_debug_sigprint();

  typedef typename T1::elem_type            eT;
  typedef typename get_pod_type<eT>::result  T;

  const SpProxy<T1> P(X.get_ref());

  const uword n_elem    = P.get_n_elem();
  const uword n_nonzero = P.get_n_nonzero();

  arma_debug_check( (n_elem == 0), "min(): given object has no elements");

   T min_val = priv::most_pos<T>();
  eT ret_val;

  if(SpProxy<T1>::must_use_iterator == true)
    {
    // We have to iterate over the elements.
    typedef typename SpProxy<T1>::const_iterator_type it_type;
    
    it_type it     = P.begin();
    it_type it_end = P.end();
    
    while (it != it_end)
      {
      const T tmp_val = std::abs(*it);
      
      if (tmp_val < min_val)
        {
        min_val = tmp_val;
        ret_val = *it;
        }
      
      ++it;
      }
    }
  else
    {
    // We can do direct access of the values, row_indices, and col_ptrs.
    // We don't need the location of the min value, so we can just call out to
    // other functions...
    ret_val = op_min::direct_min(P.get_values(), n_nonzero);
    min_val = std::abs(ret_val);
    }

  if(n_elem == n_nonzero)
    {
    return ret_val;
    }
  else
    {
    if (T(0) < min_val)
      return eT(0);
    else
      return ret_val;
    }
  }



template<typename T1>
inline
typename arma_cx_only<typename T1::elem_type>::result
spop_min::min_with_index(const SpProxy<T1>& P, uword& index_of_min_val)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type            eT;
  typedef typename get_pod_type<eT>::result  T;
  
  const uword n_elem    = P.get_n_elem();
  const uword n_nonzero = P.get_n_nonzero();
  const uword n_rows    = P.get_n_rows();
  
  arma_debug_check( (n_elem == 0), "min(): given object has no elements");
  
  T min_val = priv::most_pos<T>();
  
  if(SpProxy<T1>::must_use_iterator == true)
    {
    // We have to iterate over the elements.
    typedef typename SpProxy<T1>::const_iterator_type it_type;
    
    it_type it     = P.begin();
    it_type it_end = P.end();
    
    while (it != it_end)
      {
      const T tmp_val = std::abs(*it);
      
      if (tmp_val < min_val)
        {
                 min_val = tmp_val;
        index_of_min_val = it.row() + it.col() * n_rows;
        }
      
      ++it;
      }
    }
  else
    {
    // We can do direct access.
    min_val = std::abs(op_min::direct_min(P.get_values(), n_nonzero, index_of_min_val));

    // Convert to actual position in matrix.
    const uword row = P.get_row_indices()[index_of_min_val];
    uword col = 0;
    while (P.get_col_ptrs()[++col] < index_of_min_val + 1) { }
    index_of_min_val = (col - 1) * n_rows + row;
    }
  
  
  if(n_elem != n_nonzero)
    {
    min_val = std::min(T(0), min_val);

    // If the min_val is a nonzero element, we need its actual position in the matrix.
    if(min_val == T(0))
      {
      // Find first zero element.
      uword last_row = 0;
      uword last_col = 0;
      
      typedef typename SpProxy<T1>::const_iterator_type it_type;
      
      it_type it     = P.begin();
      it_type it_end = P.end();
      
      while (it != it_end)
        {
        // Have we moved more than one position from the last place?
        if ((it.col() == last_col) && (it.row() - last_row > 1))
          {
          index_of_min_val = it.col() * n_rows + last_row + 1;
          break;
          }
        else if ((it.col() >= last_col + 1) && (last_row < n_rows - 1))
          {
          index_of_min_val = last_col * n_rows + last_row + 1;
          break;
          }
        else if ((it.col() == last_col + 1) && (it.row() > 0))
          {
          index_of_min_val = it.col() * n_rows;
          break;
          }
        else if (it.col() > last_col + 1)
          {
          index_of_min_val = (last_col + 1) * n_rows;
          break;
          }

        last_row = it.row();
        last_col = it.col();
        ++it;
        }
      }
    }

  return P[index_of_min_val];
  }



//! @}
