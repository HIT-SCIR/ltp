// Copyright (C) 2011-2012 Ryan Curtin
// Copyright (C) 2012 Conrad Sanderson
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! \addtogroup SpValProxy
//! @{

//! SpValProxy implementation.
template<typename T1>
arma_inline
SpValProxy<T1>::SpValProxy(uword in_row, uword in_col, T1& in_parent, eT* in_val_ptr)
  : row(in_row)
  , col(in_col)
  , val_ptr(in_val_ptr)
  , parent(in_parent)
  {
  // Nothing to do.
  }



template<typename T1>
arma_inline
SpValProxy<T1>&
SpValProxy<T1>::operator=(const SpValProxy<T1>& rhs)
  {
  return (*this).operator=(eT(rhs));
  }



template<typename T1>
template<typename T2>
arma_inline
SpValProxy<T1>&
SpValProxy<T1>::operator=(const SpValProxy<T2>& rhs)
  {
  return (*this).operator=(eT(rhs));
  }



template<typename T1>
arma_inline
SpValProxy<T1>&
SpValProxy<T1>::operator=(const eT rhs)
  {
  if (rhs != eT(0)) // A nonzero element is being assigned.
    {

    if (val_ptr)
      {
      // The value exists and merely needs to be updated.
      *val_ptr = rhs;
      }

    else
      {
      // The value is nonzero and must be added.
      val_ptr = &parent.add_element(row, col, rhs);
      }

    }
  else // A zero is being assigned.~
    {

    if (val_ptr)
      {
      // The element exists, but we need to remove it, because it is being set to 0.
      parent.delete_element(row, col);
      val_ptr = NULL;
      }

    // If the element does not exist, we do not need to do anything at all.

    }

  return *this;
  }



template<typename T1>
arma_inline
SpValProxy<T1>&
SpValProxy<T1>::operator+=(const eT rhs)
  {
  if (val_ptr)
    {
    // The value already exists and merely needs to be updated.
    *val_ptr += rhs;
    check_zero();
    }
  else
    {
    if (rhs != eT(0))
      {
      // The value does not exist and must be added.
      val_ptr = &parent.add_element(row, col, rhs);
      }
    }
  
  return *this;
  }



template<typename T1>
arma_inline
SpValProxy<T1>&
SpValProxy<T1>::operator-=(const eT rhs)
  {
  if (val_ptr)
    {
    // The value already exists and merely needs to be updated.
    *val_ptr -= rhs;
    check_zero();
    }
  else
    {
    if (rhs != eT(0))
      {
      // The value does not exist and must be added.
      val_ptr = &parent.add_element(row, col, -rhs);
      }
    }

  return *this;
  }



template<typename T1>
arma_inline
SpValProxy<T1>&
SpValProxy<T1>::operator*=(const eT rhs)
  {
  if (rhs != eT(0))
    {

    if (val_ptr)
      {
      // The value already exists and merely needs to be updated.
      *val_ptr *= rhs;
      check_zero();
      }

    }
  else
    {

    if (val_ptr)
      {
      // Since we are multiplying by zero, the value can be deleted.
      parent.delete_element(row, col);
      val_ptr = NULL;
      }

    }

  return *this;
  }



template<typename T1>
arma_inline
SpValProxy<T1>&
SpValProxy<T1>::operator/=(const eT rhs)
  {
  if (rhs != eT(0)) // I hope this is true!
    {

    if (val_ptr)
      {
      *val_ptr /= rhs;
      check_zero();
      }

    }
  else
    {

    if (val_ptr)
      {
      *val_ptr /= rhs; // That is where it gets ugly.
      // Now check if it's 0.
      if (*val_ptr == eT(0))
        {
        parent.delete_element(row, col);
        val_ptr = NULL;
        }
      }

    else
      {
      eT val = eT(0) / rhs; // This may vary depending on type and implementation.

      if (val != eT(0))
        {
        // Ok, now we have to add it.
        val_ptr = &parent.add_element(row, col, val);
        }

      }
    }

  return *this;
  }



template<typename T1>
arma_inline
SpValProxy<T1>&
SpValProxy<T1>::operator++()
  {
  if (val_ptr)
    {
    (*val_ptr) += eT(1);
    check_zero();
    }

  else
    {
    val_ptr = &parent.add_element(row, col, eT(1));
    }

  return *this;
  }



template<typename T1>
arma_inline
SpValProxy<T1>&
SpValProxy<T1>::operator--()
  {
  if (val_ptr)
    {
    (*val_ptr) -= eT(1);
    check_zero();
    }

  else
    {
    val_ptr = &parent.add_element(row, col, eT(-1));
    }

  return *this;
  }



template<typename T1>
arma_inline
typename T1::elem_type
SpValProxy<T1>::operator++(const int)
  {
  if (val_ptr)
    {
    (*val_ptr) += eT(1);
    check_zero();
    }

  else
    {
    val_ptr = &parent.add_element(row, col, eT(1));
    }

  if (val_ptr) // It may have changed to now be 0.
    {
    return *(val_ptr) - eT(1);
    }
  else
    {
    return eT(0);
    }
  }



template<typename T1>
arma_inline
typename T1::elem_type
SpValProxy<T1>::operator--(const int)
  {
  if (val_ptr)
    {
    (*val_ptr) -= eT(1);
    check_zero();
    }

  else
    {
    val_ptr = &parent.add_element(row, col, eT(-1));
    }

  if (val_ptr) // It may have changed to now be 0.
    {
    return *(val_ptr) + eT(1);
    }
  else
    {
    return eT(0);
    }
  }



template<typename T1>
arma_inline
SpValProxy<T1>::operator eT() const
  {
  if (val_ptr)
    {
    return *val_ptr;
    }
  else
    {
    return eT(0);
    }
  }



template<typename T1>
arma_inline
arma_hot
void
SpValProxy<T1>::check_zero()
  {
  if (*val_ptr == eT(0))
    {
    parent.delete_element(row, col);
    val_ptr = NULL;
    }
  }



//! @}
