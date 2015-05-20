// Copyright (C) 2011-2012 Ryan Curtin
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! \addtogroup SpValProxy
//! @{

/**
 * Sparse value proxy class, meant to prevent 0s from being added to sparse
 * matrices.  T1 should be either SpMat or SpSubview, and if it's not, bad news
 * is probably coming.  This class only uses T1::add_element() and
 * T1::delete_element().
 */
template<typename T1>
class SpValProxy
  {
  public:

  typedef typename T1::elem_type eT; // Convenience typedef

  friend class SpMat<eT>;
  friend class SpSubview<eT>;
  
  /**
   * Create the sparse value proxy.
   * Otherwise, pass a pointer to a reference of the value.
   */
  arma_inline SpValProxy(uword row, uword col, T1& in_parent, eT* in_val_ptr = NULL);
  
  //! For swapping operations.
  arma_inline SpValProxy& operator=(const SpValProxy& rhs);
  template<typename T2>
  arma_inline SpValProxy& operator=(const SpValProxy<T2>& rhs);
  
  //! Overload all of the potential operators.
  
  //! First, the ones that could modify a value.
  arma_inline SpValProxy& operator=(const eT rhs);
  arma_inline SpValProxy& operator+=(const eT rhs);
  arma_inline SpValProxy& operator-=(const eT rhs);
  arma_inline SpValProxy& operator*=(const eT rhs);
  arma_inline SpValProxy& operator/=(const eT rhs);
  
  arma_inline SpValProxy& operator++();
  arma_inline SpValProxy& operator--();
  arma_inline eT operator++(const int);
  arma_inline eT operator--(const int);
  
  //! This will work for any other operations that do not modify a value.
  arma_inline operator eT() const;
  
  
  private:
  
  // Deletes the element if it is zero.  Does not check if val_ptr == NULL!
  arma_inline arma_hot void check_zero();
  
  uword row;
  uword col;
  
  eT* val_ptr;
  
  T1& parent; // We will call this object if we need to insert or delete an element.
  };



//! @}
