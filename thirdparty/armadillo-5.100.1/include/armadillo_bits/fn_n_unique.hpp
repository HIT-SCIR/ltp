// Copyright (C) 2012 Ryan Curtin
// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_n_unique
//! @{


//! \brief
//! Get the number of unique nonzero elements in two sparse matrices.
//! This is very useful for determining the amount of memory necessary before
//! a sparse matrix operation on two matrices.

template<typename T1, typename T2, typename op_n_unique_type>
inline
uword
n_unique
  (
  const SpBase<typename T1::elem_type, T1>& x,
  const SpBase<typename T2::elem_type, T2>& y,
  const op_n_unique_type junk
  )
  {
  arma_extra_debug_sigprint();
  
  const SpProxy<T1> pa(x.get_ref());
  const SpProxy<T2> pb(y.get_ref());
  
  return n_unique(pa,pb,junk);
  }



template<typename T1, typename T2, typename op_n_unique_type>
arma_hot
inline
uword
n_unique
  (
  const SpProxy<T1>& pa,
  const SpProxy<T2>& pb,
  const op_n_unique_type junk
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typename SpProxy<T1>::const_iterator_type x_it     = pa.begin();
  typename SpProxy<T1>::const_iterator_type x_it_end = pa.end();
  
  typename SpProxy<T2>::const_iterator_type y_it     = pb.begin();
  typename SpProxy<T2>::const_iterator_type y_it_end = pb.end();

  uword total_n_nonzero = 0;

  while( (x_it != x_it_end) || (y_it != y_it_end) )
    {
    if(x_it == y_it)
      {
      if(op_n_unique_type::eval((*x_it), (*y_it)) != typename T1::elem_type(0))
        {
        ++total_n_nonzero;
        }

      ++x_it;
      ++y_it;
      }
    else
      {
      if((x_it.col() < y_it.col()) || ((x_it.col() == y_it.col()) && (x_it.row() < y_it.row()))) // if y is closer to the end
        {
        if(op_n_unique_type::eval((*x_it), typename T1::elem_type(0)) != typename T1::elem_type(0))
          {
          ++total_n_nonzero;
          }

        ++x_it;
        }
      else // x is closer to the end
        {
        if(op_n_unique_type::eval(typename T1::elem_type(0), (*y_it)) != typename T1::elem_type(0))
          {
          ++total_n_nonzero;
          }

        ++y_it;
        }
      }
    }

  return total_n_nonzero;
  }


// Simple operators.
struct op_n_unique_add
  {
  template<typename eT> inline static eT eval(const eT& l, const eT& r) { return (l + r); }
  };

struct op_n_unique_sub
  {
  template<typename eT> inline static eT eval(const eT& l, const eT& r) { return (l - r); }
  };

struct op_n_unique_mul
  {
  template<typename eT> inline static eT eval(const eT& l, const eT& r) { return (l * r); }
  };

struct op_n_unique_count
  {
  template<typename eT> inline static eT eval(const eT&, const eT&) { return eT(1); }
  };



//! @}
