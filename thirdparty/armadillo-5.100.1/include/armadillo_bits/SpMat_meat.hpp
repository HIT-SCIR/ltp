// Copyright (C) 2011-2015 Ryan Curtin
// Copyright (C) 2012-2015 Conrad Sanderson
// Copyright (C) 2011 Matthew Amidon
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! \addtogroup SpMat
//! @{

/**
 * Initialize a sparse matrix with size 0x0 (empty).
 */
template<typename eT>
inline
SpMat<eT>::SpMat()
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(memory::acquire_chunked<eT>(1))
  , row_indices(memory::acquire_chunked<uword>(1))
  , col_ptrs(memory::acquire<uword>(2))
  {
  arma_extra_debug_sigprint_this(this);
  
  access::rw(values[0]) = 0;
  access::rw(row_indices[0]) = 0;
  
  access::rw(col_ptrs[0]) = 0; // No elements.
  access::rw(col_ptrs[1]) = std::numeric_limits<uword>::max();
  }



/**
 * Clean up the memory of a sparse matrix and destruct it.
 */
template<typename eT>
inline
SpMat<eT>::~SpMat()
  {
  arma_extra_debug_sigprint_this(this);
  
  if(values     )  { memory::release(access::rw(values));      }
  if(row_indices)  { memory::release(access::rw(row_indices)); }
  if(col_ptrs   )  { memory::release(access::rw(col_ptrs));    }
  }



/**
 * Constructor with size given.
 */
template<typename eT>
inline
SpMat<eT>::SpMat(const uword in_rows, const uword in_cols)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL)
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);

  init(in_rows, in_cols);
  }



/**
 * Assemble from text.
 */
template<typename eT>
inline
SpMat<eT>::SpMat(const char* text)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL)
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);

  init(std::string(text));
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const char* text)
  {
  arma_extra_debug_sigprint();

  init(std::string(text));
  
  return *this;
  }



template<typename eT>
inline
SpMat<eT>::SpMat(const std::string& text)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL)
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint();

  init(text);
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const std::string& text)
  {
  arma_extra_debug_sigprint();
  
  init(text);
  
  return *this;
  }



template<typename eT>
inline
SpMat<eT>::SpMat(const SpMat<eT>& x)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL)
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);

  init(x);
  }



#if defined(ARMA_USE_CXX11)
  
  template<typename eT>
  inline
  SpMat<eT>::SpMat(SpMat<eT>&& in_mat)
    : n_rows(0)
    , n_cols(0)
    , n_elem(0)
    , n_nonzero(0)
    , vec_state(0)
    , values(NULL)
    , row_indices(NULL)
    , col_ptrs(NULL)
    {
    arma_extra_debug_sigprint_this(this);
    arma_extra_debug_sigprint(arma_boost::format("this = %x   in_mat = %x") % this % &in_mat);
    
    (*this).steal_mem(in_mat);
    }
  
  
  
  template<typename eT>
  inline
  const SpMat<eT>&
  SpMat<eT>::operator=(SpMat<eT>&& in_mat)
    {
    arma_extra_debug_sigprint(arma_boost::format("this = %x   in_mat = %x") % this % &in_mat);
    
    (*this).steal_mem(in_mat);
    
    return *this;
    }
  
#endif



//! Insert a large number of values at once.
//! locations.row[0] should be row indices, locations.row[1] should be column indices,
//! and values should be the corresponding values.
//! If sort_locations is false, then it is assumed that the locations and values
//! are already sorted in column-major ordering.
template<typename eT>
template<typename T1, typename T2>
inline
SpMat<eT>::SpMat(const Base<uword,T1>& locations_expr, const Base<eT,T2>& vals_expr, const bool sort_locations)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL)
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);
  
  const unwrap<T1> locs_tmp( locations_expr.get_ref() );
  const unwrap<T2> vals_tmp(      vals_expr.get_ref() );
  
  const Mat<uword>& locs = locs_tmp.M;
  const Mat<eT>&    vals = vals_tmp.M;
  
  arma_debug_check( (vals.is_vec() == false),     "SpMat::SpMat(): given 'values' object is not a vector"                  );
  arma_debug_check( (locs.n_rows != 2),           "SpMat::SpMat(): locations matrix must have two rows"                    );
  arma_debug_check( (locs.n_cols != vals.n_elem), "SpMat::SpMat(): number of locations is different than number of values" );

  // If there are no elements in the list, max() will fail.
  if(locs.n_cols == 0)  { init(0, 0); return; }
  
  // Automatically determine size before pruning zeros.
  uvec bounds = arma::max(locs, 1);
  init(bounds[0] + 1, bounds[1] + 1);
  
  // Ensure that there are no zeros
  const uword N_old = vals.n_elem;
        uword N_new = 0;
  
  for(uword i = 0; i < N_old; ++i)
    {
    if(vals[i] != eT(0))  { ++N_new; }
    }
  
  if(N_new != N_old)
    {
    Col<eT>    filtered_vals(N_new);
    Mat<uword> filtered_locs(2, N_new);
    
    uword index = 0;
    for(uword i = 0; i < N_old; ++i)
      {
      if(vals[i] != eT(0))
        {
        filtered_vals[index] = vals[i];
        
        filtered_locs.at(0, index) = locs.at(0, i);
        filtered_locs.at(1, index) = locs.at(1, i);
        
        ++index;
        }
      }
    
    init_batch_std(filtered_locs, filtered_vals, sort_locations);
    }
  else
    {
    init_batch_std(locs, vals, sort_locations);
    }
  }



//! Insert a large number of values at once.
//! locations.row[0] should be row indices, locations.row[1] should be column indices,
//! and values should be the corresponding values.
//! If sort_locations is false, then it is assumed that the locations and values
//! are already sorted in column-major ordering.
//! In this constructor the size is explicitly given.
template<typename eT>
template<typename T1, typename T2>
inline
SpMat<eT>::SpMat(const Base<uword,T1>& locations_expr, const Base<eT,T2>& vals_expr, const uword in_n_rows, const uword in_n_cols, const bool sort_locations, const bool check_for_zeros)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL)
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);
  
  const unwrap<T1> locs_tmp( locations_expr.get_ref() );
  const unwrap<T2> vals_tmp(      vals_expr.get_ref() );
  
  const Mat<uword>& locs = locs_tmp.M;
  const Mat<eT>&    vals = vals_tmp.M;
  
  arma_debug_check( (vals.is_vec() == false),     "SpMat::SpMat(): given 'values' object is not a vector"                  );
  arma_debug_check( (locs.n_rows != 2),           "SpMat::SpMat(): locations matrix must have two rows"                    );
  arma_debug_check( (locs.n_cols != vals.n_elem), "SpMat::SpMat(): number of locations is different than number of values" );
  
  init(in_n_rows, in_n_cols);

  // Ensure that there are no zeros, unless the user asked not to.
  if(check_for_zeros)
    {
    const uword N_old = vals.n_elem;
          uword N_new = 0;
    
    for(uword i = 0; i < N_old; ++i)
      {
      if(vals[i] != eT(0))  { ++N_new; }
      }
    
    if(N_new != N_old)
      {
      Col<eT>    filtered_vals(N_new);
      Mat<uword> filtered_locs(2, N_new);
      
      uword index = 0;
      for(uword i = 0; i < N_old; ++i)
        {
        if(vals[i] != eT(0))
          {
          filtered_vals[index] = vals[i];
          
          filtered_locs.at(0, index) = locs.at(0, i);
          filtered_locs.at(1, index) = locs.at(1, i);
          
          ++index;
          }
        }
      
      init_batch_std(filtered_locs, filtered_vals, sort_locations);
      }
    else
      {
      init_batch_std(locs, vals, sort_locations);
      }
    }
  else
    {
    init_batch_std(locs, vals, sort_locations);
    }
  }



template<typename eT>
template<typename T1, typename T2>
inline
SpMat<eT>::SpMat(const bool add_values, const Base<uword,T1>& locations_expr, const Base<eT,T2>& vals_expr, const uword in_n_rows, const uword in_n_cols, const bool sort_locations, const bool check_for_zeros)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL)
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);
  
  const unwrap<T1> locs_tmp( locations_expr.get_ref() );
  const unwrap<T2> vals_tmp(      vals_expr.get_ref() );
  
  const Mat<uword>& locs = locs_tmp.M;
  const Mat<eT>&    vals = vals_tmp.M;
  
  arma_debug_check( (vals.is_vec() == false),     "SpMat::SpMat(): given 'values' object is not a vector"                  );
  arma_debug_check( (locs.n_rows != 2),           "SpMat::SpMat(): locations matrix must have two rows"                    );
  arma_debug_check( (locs.n_cols != vals.n_elem), "SpMat::SpMat(): number of locations is different than number of values" );
  
  init(in_n_rows, in_n_cols);

  // Ensure that there are no zeros, unless the user asked not to.
  if(check_for_zeros)
    {
    const uword N_old = vals.n_elem;
          uword N_new = 0;
    
    for(uword i = 0; i < N_old; ++i)
      {
      if(vals[i] != eT(0))  { ++N_new; }
      }
    
    if(N_new != N_old)
      {
      Col<eT>    filtered_vals(N_new);
      Mat<uword> filtered_locs(2, N_new);
      
      uword index = 0;
      for(uword i = 0; i < N_old; ++i)
        {
        if(vals[i] != eT(0))
          {
          filtered_vals[index] = vals[i];
          
          filtered_locs.at(0, index) = locs.at(0, i);
          filtered_locs.at(1, index) = locs.at(1, i);
          
          ++index;
          }
        }
      
      add_values ? init_batch_add(filtered_locs, filtered_vals, sort_locations) : init_batch_std(filtered_locs, filtered_vals, sort_locations);
      }
    else
      {
      add_values ? init_batch_add(locs, vals, sort_locations) : init_batch_std(locs, vals, sort_locations);
      }
    }
  else
    {
    add_values ? init_batch_add(locs, vals, sort_locations) : init_batch_std(locs, vals, sort_locations);
    }
  }



//! Insert a large number of values at once.
//! Per CSC format, rowind_expr should be row indices, 
//! colptr_expr should column ptr indices locations,
//! and values should be the corresponding values.
//! In this constructor the size is explicitly given.
//! Values are assumed to be sorted, and the size 
//! information is trusted
template<typename eT>
template<typename T1, typename T2, typename T3>
inline
SpMat<eT>::SpMat
  (
  const Base<uword,T1>& rowind_expr, 
  const Base<uword,T2>& colptr_expr, 
  const Base<eT,   T3>& values_expr, 
  const uword           in_n_rows, 
  const uword           in_n_cols
  )
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL)
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);
  
  init(in_n_rows, in_n_cols);
  
  const unwrap<T1> rowind_tmp( rowind_expr.get_ref() );
  const unwrap<T2> colptr_tmp( colptr_expr.get_ref() );
  const unwrap<T3>   vals_tmp( values_expr.get_ref() );
  
  const Mat<uword>& rowind = rowind_tmp.M;
  const Mat<uword>& colptr = colptr_tmp.M;
  const Mat<eT>&      vals = vals_tmp.M;
  
  arma_debug_check( (rowind.is_vec() == false), "SpMat::SpMat(): given 'rowind' object is not a vector" );
  arma_debug_check( (colptr.is_vec() == false), "SpMat::SpMat(): given 'colptr' object is not a vector" );
  arma_debug_check( (vals.is_vec()   == false), "SpMat::SpMat(): given 'values' object is not a vector" );
  
  arma_debug_check( (rowind.n_elem != vals.n_elem), "SpMat::SpMat(): number of row indices is not equal to number of values" );
  arma_debug_check( (colptr.n_elem != (n_cols+1) ), "SpMat::SpMat(): number of column pointers is not equal to n_cols+1" );
  
  // Resize to correct number of elements (this also sets n_nonzero)
  mem_resize(vals.n_elem);
  
  // copy supplied values into sparse matrix -- not checked for consistency
  arrayops::copy(access::rwp(row_indices), rowind.memptr(), rowind.n_elem );
  arrayops::copy(access::rwp(col_ptrs),    colptr.memptr(), colptr.n_elem );
  arrayops::copy(access::rwp(values),      vals.memptr(),   vals.n_elem   );
  
  // important: set the sentinel as well
  access::rw(col_ptrs[n_cols + 1]) = std::numeric_limits<uword>::max();
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const eT val)
  {
  arma_extra_debug_sigprint();
  
  if(val != eT(0))
    {
    // Resize to 1x1 then set that to the right value.
    init(1, 1); // Sets col_ptrs to 0.
    mem_resize(1); // One element.
    
    // Manually set element.
    access::rw(values[0]) = val;
    access::rw(row_indices[0]) = 0;
    access::rw(col_ptrs[1]) = 1;
    }
  else
    {
    init(0, 0);
    }
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const eT val)
  {
  arma_extra_debug_sigprint();
  
  if(val != eT(0))
    {
    arrayops::inplace_mul( access::rwp(values), val, n_nonzero );
    
    remove_zeros();
    }
  else
    {
    // Everything will be zero.
    init(n_rows, n_cols);
    }
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const eT val)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (val == eT(0)), "element-wise division: division by zero" );
  
  arrayops::inplace_div( access::rwp(values), val, n_nonzero );
  
  remove_zeros();
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  init(x);
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT> out = (*this) + x;
  
  steal_mem(out);
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT> out = (*this) - x;
  
  steal_mem(out);
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const SpMat<eT>& y)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT> z = (*this) * y;
  
  steal_mem(z);
  
  return *this;
  }



// This is in-place element-wise matrix multiplication.
template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const SpMat<eT>& y)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT> z = (*this) % y;
  
  steal_mem(z);
  
  return *this;
  }



// Construct a complex matrix out of two non-complex matrices
template<typename eT>
template<typename T1, typename T2>
inline
SpMat<eT>::SpMat
  (
  const SpBase<typename SpMat<eT>::pod_type, T1>& A,
  const SpBase<typename SpMat<eT>::pod_type, T2>& B
  )
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL) // extra element is set when mem_resize is called
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type T;
  
  // Make sure eT is complex and T is not (compile-time check).
  arma_type_check(( is_complex<eT>::value == false ));
  arma_type_check(( is_complex< T>::value == true  ));
  
  // Compile-time abort if types are not compatible.
  arma_type_check(( is_same_type< std::complex<T>, eT >::no ));
  
  const unwrap_spmat<T1> tmp1(A.get_ref());
  const unwrap_spmat<T2> tmp2(B.get_ref());
  
  const SpMat<T>& X = tmp1.M;
  const SpMat<T>& Y = tmp2.M;
  
  arma_debug_assert_same_size(X.n_rows, X.n_cols, Y.n_rows, Y.n_cols, "SpMat()");
  
  const uword l_n_rows = X.n_rows;
  const uword l_n_cols = X.n_cols;
  
  // Set size of matrix correctly.
  init(l_n_rows, l_n_cols);
  mem_resize(n_unique(X, Y, op_n_unique_count()));
  
  // Now on a second iteration, fill it.
  typename SpMat<T>::const_iterator x_it  = X.begin();
  typename SpMat<T>::const_iterator x_end = X.end();
  
  typename SpMat<T>::const_iterator y_it  = Y.begin();
  typename SpMat<T>::const_iterator y_end = Y.end();
  
  uword cur_pos = 0;
  
  while ((x_it != x_end) || (y_it != y_end))
    {
    if(x_it == y_it) // if we are at the same place
      {
      access::rw(values[cur_pos]) = std::complex<T>((T) *x_it, (T) *y_it);
      access::rw(row_indices[cur_pos]) = x_it.row();
      ++access::rw(col_ptrs[x_it.col() + 1]);
      
      ++x_it;
      ++y_it;
      }
    else
      {
      if((x_it.col() < y_it.col()) || ((x_it.col() == y_it.col()) && (x_it.row() < y_it.row()))) // if y is closer to the end
        {
        access::rw(values[cur_pos]) = std::complex<T>((T) *x_it, T(0));
        access::rw(row_indices[cur_pos]) = x_it.row();
        ++access::rw(col_ptrs[x_it.col() + 1]);
        
        ++x_it;
        }
      else // x is closer to the end
        {
        access::rw(values[cur_pos]) = std::complex<T>(T(0), (T) *y_it);
        access::rw(row_indices[cur_pos]) = y_it.row();
        ++access::rw(col_ptrs[y_it.col() + 1]);
        
        ++y_it;
        }
      }
    
    ++cur_pos;
    }
  
  // Now fix the column pointers; they are supposed to be a sum.
  for (uword c = 1; c <= n_cols; ++c)
    {
    access::rw(col_ptrs[c]) += col_ptrs[c - 1];
    }
  
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(n_rows, n_cols, x.n_rows, x.n_cols, "element-wise division");
  
  // If you use this method, you are probably stupid or misguided, but for compatibility with Mat, we have implemented it anyway.
  // We have to loop over every element, which is not good.  In fact, it makes me physically sad to write this.
  for(uword c = 0; c < n_cols; ++c)
    {
    for(uword r = 0; r < n_rows; ++r)
      {
      at(r, c) /= x.at(r, c);
      }
    }

  return *this;
  }



template<typename eT>
template<typename T1>
inline
SpMat<eT>::SpMat(const Base<eT, T1>& x)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL) // extra element is set when mem_resize is called in operator=()
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);

  (*this).operator=(x);
  }



template<typename eT>
template<typename T1>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const Base<eT, T1>& expr)
  {
  arma_extra_debug_sigprint();
  
  const quasi_unwrap<T1> tmp(expr.get_ref());
  const Mat<eT>& x     = tmp.M;
  
  const uword x_n_rows = x.n_rows;
  const uword x_n_cols = x.n_cols;
  const uword x_n_elem = x.n_elem;
  
  init(x_n_rows, x_n_cols);
  
  // Count number of nonzero elements in base object.
  uword n = 0;
  
  const eT* x_mem = x.memptr();
  
  for(uword i = 0; i < x_n_elem; ++i)
    {
    n += (x_mem[i] != eT(0)) ? uword(1) : uword(0);
    }
  
  mem_resize(n);
  
  // Now the memory is resized correctly; add nonzero elements.
  n = 0;
  for(uword j = 0; j < x_n_cols; ++j)
  for(uword i = 0; i < x_n_rows; ++i)
    {
    const eT val = (*x_mem);  x_mem++;
    
    if(val != eT(0))
      {
      access::rw(values[n])      = val;
      access::rw(row_indices[n]) = i;
      access::rw(col_ptrs[j + 1])++;
      ++n;
      }
    }
  
  // Sum column counts to be column pointers.
  for(uword c = 1; c <= n_cols; ++c)
    {
    access::rw(col_ptrs[c]) += col_ptrs[c - 1];
    }
  
  return *this;
  }



template<typename eT>
template<typename T1>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const Base<eT, T1>& x)
  {
  arma_extra_debug_sigprint();
  
  return (*this).operator=( (*this) + x.get_ref() );
  }



template<typename eT>
template<typename T1>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const Base<eT, T1>& x)
  {
  arma_extra_debug_sigprint();
  
  return (*this).operator=( (*this) - x.get_ref() );
  }



template<typename eT>
template<typename T1>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const Base<eT, T1>& y)
  {
  arma_extra_debug_sigprint();

  const Proxy<T1> p(y.get_ref());

  arma_debug_assert_mul_size(n_rows, n_cols, p.get_n_rows(), p.get_n_cols(), "matrix multiplication");

  // We assume the matrix structure is such that we will end up with a sparse
  // matrix.  Assuming that every entry in the dense matrix is nonzero (which is
  // a fairly valid assumption), each row with any nonzero elements in it (in this
  // matrix) implies an entire nonzero column.  Therefore, we iterate over all
  // the row_indices and count the number of rows with any elements in them
  // (using the quasi-linked-list idea from SYMBMM -- see operator_times.hpp).
  podarray<uword> index(n_rows);
  index.fill(n_rows); // Fill with invalid links.

  uword last_index = n_rows + 1;
  for(uword i = 0; i < n_nonzero; ++i)
    {
    if(index[row_indices[i]] == n_rows)
      {
      index[row_indices[i]] = last_index;
      last_index = row_indices[i];
      }
    }

  // Now count the number of rows which have nonzero elements.
  uword nonzero_rows = 0;
  while(last_index != n_rows + 1)
    {
    ++nonzero_rows;
    last_index = index[last_index];
    }

  SpMat<eT> z(n_rows, p.get_n_cols());

  z.mem_resize(nonzero_rows * p.get_n_cols()); // upper bound on size

  // Now we have to fill all the elements using a modification of the NUMBMM algorithm.
  uword cur_pos = 0;

  podarray<eT> partial_sums(n_rows);
  partial_sums.zeros();

  for(uword lcol = 0; lcol < n_cols; ++lcol)
    {
    const_iterator it = begin();

    while(it != end())
      {
      const eT value = (*it);

      partial_sums[it.row()] += (value * p.at(it.col(), lcol));

      ++it;
      }

    // Now add all partial sums to the matrix.
    for(uword i = 0; i < n_rows; ++i)
      {
      if(partial_sums[i] != eT(0))
        {
        access::rw(z.values[cur_pos]) = partial_sums[i];
        access::rw(z.row_indices[cur_pos]) = i;
        ++access::rw(z.col_ptrs[lcol + 1]);
        //printf("colptr %d now %d\n", lcol + 1, z.col_ptrs[lcol + 1]);
        ++cur_pos;
        partial_sums[i] = 0; // Would it be faster to do this in batch later?
        }
      }
    }

  // Now fix the column pointers.
  for(uword c = 1; c <= z.n_cols; ++c)
    {
    access::rw(z.col_ptrs[c]) += z.col_ptrs[c - 1];
    }

  // Resize to final correct size.
  z.mem_resize(z.col_ptrs[z.n_cols]);
  
  // Now take the memory of the temporary matrix.
  steal_mem(z);
  
  return *this;
  }



/**
 * Don't use this function.  It's not mathematically well-defined and wastes
 * cycles to trash all your data.  This is dumb.
 */
template<typename eT>
template<typename T1>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const Base<eT, T1>& x)
  {
  arma_extra_debug_sigprint();

  SpMat<eT> tmp = (*this) / x.get_ref();
  
  steal_mem(tmp);
  
  return *this;
  }



template<typename eT>
template<typename T1>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const Base<eT, T1>& x)
  {
  arma_extra_debug_sigprint();

  const Proxy<T1> p(x.get_ref());
  
  arma_debug_assert_same_size(n_rows, n_cols, p.get_n_rows(), p.get_n_cols(), "element-wise multiplication");
  
  // Count the number of elements we will need.
  SpMat<eT> tmp(n_rows, n_cols);
  const_iterator it = begin();
  uword new_n_nonzero = 0;

  while(it != end())
    {
    // prefer_at_accessor == false can't save us any work here
    if(((*it) * p.at(it.row(), it.col())) != eT(0))
      {
      ++new_n_nonzero;
      }
    ++it;
    }

  // Resize.
  tmp.mem_resize(new_n_nonzero);

  const_iterator c_it = begin();
  uword cur_pos = 0;
  while(c_it != end())
    {
    // prefer_at_accessor == false can't save us any work here
    const eT val = (*c_it) * p.at(c_it.row(), c_it.col());
    if(val != eT(0))
      {
      access::rw(tmp.values[cur_pos]) = val;
      access::rw(tmp.row_indices[cur_pos]) = c_it.row();
      ++access::rw(tmp.col_ptrs[c_it.col() + 1]);
      ++cur_pos;
      }

    ++c_it;
    }

  // Fix column pointers.
  for(uword c = 1; c <= n_cols; ++c)
    {
    access::rw(tmp.col_ptrs[c]) += tmp.col_ptrs[c - 1];
    }

  steal_mem(tmp);

  return *this;
  }



/**
 * Functions on subviews.
 */
template<typename eT>
inline
SpMat<eT>::SpMat(const SpSubview<eT>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL) // extra element added when mem_resize is called
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const SpSubview<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  const uword in_n_cols = X.n_cols;
  const uword in_n_rows = X.n_rows;
  
  const bool alias = (this == &(X.m));

  if(alias == false)
    {
    init(in_n_rows, in_n_cols);

    const uword x_n_nonzero = X.n_nonzero;

    mem_resize(x_n_nonzero);

    typename SpSubview<eT>::const_iterator it = X.begin();

    while(it != X.end())
      {
      access::rw(row_indices[it.pos()]) = it.row();
      access::rw(values[it.pos()]) = (*it);
      ++access::rw(col_ptrs[it.col() + 1]);
      ++it;
      }

    // Now sum column pointers.
    for(uword c = 1; c <= n_cols; ++c)
      {
      access::rw(col_ptrs[c]) += col_ptrs[c - 1];
      }
    }
  else
    {
    // Create it in a temporary.
    SpMat<eT> tmp(X);

    steal_mem(tmp);
    }
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const SpSubview<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT> tmp = (*this) + X;
  
  steal_mem(tmp);
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const SpSubview<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT> tmp = (*this) - X;
  
  steal_mem(tmp);
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const SpSubview<eT>& y)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT> z = (*this) * y;
  
  steal_mem(z);
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const SpSubview<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  SpMat<eT> tmp = (*this) % x;
  
  steal_mem(tmp);
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const SpSubview<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_assert_same_size(n_rows, n_cols, x.n_rows, x.n_cols, "element-wise division");
  
  // There is no pretty way to do this.
  for(uword elem = 0; elem < n_elem; elem++)
    {
    at(elem) /= x(elem);
    }
  
  return *this;
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
SpMat<eT>::SpMat(const SpOp<T1, spop_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL) // set in application of sparse operation
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  spop_type::apply(*this, X);
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const SpOp<T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  spop_type::apply(*this, X);
  
  return *this;
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const SpOp<T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  const SpMat<eT> m(X);
  
  return (*this).operator+=(m);
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const SpOp<T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  const SpMat<eT> m(X);
  
  return (*this).operator-=(m);
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const SpOp<T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  const SpMat<eT> m(X);
  
  return (*this).operator*=(m);
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const SpOp<T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  const SpMat<eT> m(X);
  
  return (*this).operator%=(m);
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const SpOp<T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  const SpMat<eT> m(X);
  
  return (*this).operator/=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename spglue_type>
inline
SpMat<eT>::SpMat(const SpGlue<T1, T2, spglue_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL) // extra element set in application of sparse glue
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  spglue_type::apply(*this, X);
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
SpMat<eT>::SpMat(const mtSpOp<eT, T1, spop_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem(0)
  , n_nonzero(0)
  , vec_state(0)
  , values(NULL) // extra element set in application of sparse glue
  , row_indices(NULL)
  , col_ptrs(NULL)
  {
  arma_extra_debug_sigprint_this(this);

  spop_type::apply(*this, X);
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const mtSpOp<eT, T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();

  spop_type::apply(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const mtSpOp<eT, T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();

  const SpMat<eT> m(X);

  return (*this).operator+=(m);
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const mtSpOp<eT, T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();

  const SpMat<eT> m(X);

  return (*this).operator-=(m);
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const mtSpOp<eT, T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();

  const SpMat<eT> m(X);

  return (*this).operator*=(m);
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const mtSpOp<eT, T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();

  const SpMat<eT> m(X);

  return (*this).operator%=(m);
  }



template<typename eT>
template<typename T1, typename spop_type>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const mtSpOp<eT, T1, spop_type>& X)
  {
  arma_extra_debug_sigprint();

  const SpMat<eT> m(X);

  return (*this).operator/=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename spglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator=(const SpGlue<T1, T2, spglue_type>& X)
  {
  arma_extra_debug_sigprint();

  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  spglue_type::apply(*this, X);
  
  return *this;
  }



template<typename eT>
template<typename T1, typename T2, typename spglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator+=(const SpGlue<T1, T2, spglue_type>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  const SpMat<eT> m(X);
  
  return (*this).operator+=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename spglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator-=(const SpGlue<T1, T2, spglue_type>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  const SpMat<eT> m(X);
  
  return (*this).operator-=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename spglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator*=(const SpGlue<T1, T2, spglue_type>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  const SpMat<eT> m(X);
  
  return (*this).operator*=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename spglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator%=(const SpGlue<T1, T2, spglue_type>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  const SpMat<eT> m(X);
  
  return (*this).operator%=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename spglue_type>
inline
const SpMat<eT>&
SpMat<eT>::operator/=(const SpGlue<T1, T2, spglue_type>& X)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  
  const SpMat<eT> m(X);
  
  return (*this).operator/=(m);
  }



template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::row(const uword row_num)
  {
  arma_extra_debug_sigprint();

  arma_debug_check(row_num >= n_rows, "SpMat::row(): out of bounds");

  return SpSubview<eT>(*this, row_num, 0, 1, n_cols);
  }



template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::row(const uword row_num) const
  {
  arma_extra_debug_sigprint();

  arma_debug_check(row_num >= n_rows, "SpMat::row(): out of bounds");

  return SpSubview<eT>(*this, row_num, 0, 1, n_cols);
  }



template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::operator()(const uword row_num, const span& col_span)
  {
  arma_extra_debug_sigprint();
  
  const bool col_all = col_span.whole;
  
  const uword local_n_cols = n_cols;
  
  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;
  
  arma_debug_check
    (
    (row_num >= n_rows)
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "SpMat::operator(): indices out of bounds or incorrectly used"
    );
  
  return SpSubview<eT>(*this, row_num, in_col1, 1, submat_n_cols);
  }



template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::operator()(const uword row_num, const span& col_span) const
  {
  arma_extra_debug_sigprint();
  
  const bool col_all = col_span.whole;
  
  const uword local_n_cols = n_cols;
  
  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;
  
  arma_debug_check
    (
    (row_num >= n_rows)
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "SpMat::operator(): indices out of bounds or incorrectly used"
    );
  
  return SpSubview<eT>(*this, row_num, in_col1, 1, submat_n_cols);
  }



template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::col(const uword col_num)
  {
  arma_extra_debug_sigprint();

  arma_debug_check(col_num >= n_cols, "SpMat::col(): out of bounds");

  return SpSubview<eT>(*this, 0, col_num, n_rows, 1);
  }



template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::col(const uword col_num) const
  {
  arma_extra_debug_sigprint();

  arma_debug_check(col_num >= n_cols, "SpMat::col(): out of bounds");

  return SpSubview<eT>(*this, 0, col_num, n_rows, 1);
  }



template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::operator()(const span& row_span, const uword col_num)
  {
  arma_extra_debug_sigprint();
  
  const bool row_all = row_span.whole;
  
  const uword local_n_rows = n_rows;
  
  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;
  
  arma_debug_check
    (
    (col_num >= n_cols)
    ||
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ,
    "SpMat::operator(): indices out of bounds or incorrectly used"
    );
  
  return SpSubview<eT>(*this, in_row1, col_num, submat_n_rows, 1);
  }



template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::operator()(const span& row_span, const uword col_num) const
  {
  arma_extra_debug_sigprint();
  
  const bool row_all = row_span.whole;
  
  const uword local_n_rows = n_rows;
  
  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;
  
  arma_debug_check
    (
    (col_num >= n_cols)
    ||
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ,
    "SpMat::operator(): indices out of bounds or incorrectly used"
    );
  
  return SpSubview<eT>(*this, in_row1, col_num, submat_n_rows, 1);
  }



template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::rows(const uword in_row1, const uword in_row2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "SpMat::rows(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return SpSubview<eT>(*this, in_row1, 0, subview_n_rows, n_cols);
  }



template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::rows(const uword in_row1, const uword in_row2) const
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "SpMat::rows(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return SpSubview<eT>(*this, in_row1, 0, subview_n_rows, n_cols);
  }



template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::cols(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "SpMat::cols(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return SpSubview<eT>(*this, 0, in_col1, n_rows, subview_n_cols);
  }



template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::cols(const uword in_col1, const uword in_col2) const
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "SpMat::cols(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return SpSubview<eT>(*this, 0, in_col1, n_rows, subview_n_cols);
  }



template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_row1 > in_row2) || (in_col1 >  in_col2) || (in_row2 >= n_rows) || (in_col2 >= n_cols),
    "SpMat::submat(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_rows = in_row2 - in_row1 + 1;
  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return SpSubview<eT>(*this, in_row1, in_col1, subview_n_rows, subview_n_cols);
  }



template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2) const
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_row1 > in_row2) || (in_col1 >  in_col2) || (in_row2 >= n_rows) || (in_col2 >= n_cols),
    "SpMat::submat(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_rows = in_row2 - in_row1 + 1;
  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return SpSubview<eT>(*this, in_row1, in_col1, subview_n_rows, subview_n_cols);
  }



template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::submat(const uword in_row1, const uword in_col1, const SizeMat& s)
  {
  arma_extra_debug_sigprint();
  
  const uword l_n_rows = n_rows;
  const uword l_n_cols = n_cols;
  
  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;
  
  arma_debug_check
    (
    ((in_row1 >= l_n_rows) || (in_col1 >= l_n_cols) || ((in_row1 + s_n_rows) > l_n_rows) || ((in_col1 + s_n_cols) > l_n_cols)),
    "SpMat::submat(): indices or size out of bounds"
    );
  
  return SpSubview<eT>(*this, in_row1, in_col1, s_n_rows, s_n_cols);
  }



template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::submat(const uword in_row1, const uword in_col1, const SizeMat& s) const
  {
  arma_extra_debug_sigprint();
  
  const uword l_n_rows = n_rows;
  const uword l_n_cols = n_cols;
  
  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;
  
  arma_debug_check
    (
    ((in_row1 >= l_n_rows) || (in_col1 >= l_n_cols) || ((in_row1 + s_n_rows) > l_n_rows) || ((in_col1 + s_n_cols) > l_n_cols)),
    "SpMat::submat(): indices or size out of bounds"
    );
  
  return SpSubview<eT>(*this, in_row1, in_col1, s_n_rows, s_n_cols);
  }



template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::submat(const span& row_span, const span& col_span)
  {
  arma_extra_debug_sigprint();

  const bool row_all = row_span.whole;
  const bool col_all = col_span.whole;
  
  const uword local_n_rows = n_rows;
  const uword local_n_cols = n_cols;
  
  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1; 
  
  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1; 
  
  arma_debug_check
    (    
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ||   
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,    
    "SpMat::submat(): indices out of bounds or incorrectly used"
    );   
  
  return SpSubview<eT>(*this, in_row1, in_col1, submat_n_rows, submat_n_cols);
  }



template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::submat(const span& row_span, const span& col_span) const
  {
  arma_extra_debug_sigprint();

  const bool row_all = row_span.whole;
  const bool col_all = col_span.whole;
  
  const uword local_n_rows = n_rows;
  const uword local_n_cols = n_cols;
  
  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1; 
  
  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1; 
  
  arma_debug_check
    (    
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ||   
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,    
    "SpMat::submat(): indices out of bounds or incorrectly used"
    );   
  
  return SpSubview<eT>(*this, in_row1, in_col1, submat_n_rows, submat_n_cols);
  }



template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::operator()(const span& row_span, const span& col_span)
  {
  arma_extra_debug_sigprint();

  return submat(row_span, col_span);
  }



template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::operator()(const span& row_span, const span& col_span) const
  {
  arma_extra_debug_sigprint();

  return submat(row_span, col_span);
  }



template<typename eT>
arma_inline
SpSubview<eT>
SpMat<eT>::operator()(const uword in_row1, const uword in_col1, const SizeMat& s)
  {
  arma_extra_debug_sigprint();
  
  return (*this).submat(in_row1, in_col1, s);
  }



template<typename eT>
arma_inline
const SpSubview<eT>
SpMat<eT>::operator()(const uword in_row1, const uword in_col1, const SizeMat& s) const
  {
  arma_extra_debug_sigprint();
  
  return (*this).submat(in_row1, in_col1, s);
  }



template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::head_rows(const uword N)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > n_rows), "SpMat::head_rows(): size out of bounds");
  
  return SpSubview<eT>(*this, 0, 0, N, n_cols);
  }



template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::head_rows(const uword N) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > n_rows), "SpMat::head_rows(): size out of bounds");
  
  return SpSubview<eT>(*this, 0, 0, N, n_cols);
  }



template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::tail_rows(const uword N)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > n_rows), "SpMat::tail_rows(): size out of bounds");
  
  const uword start_row = n_rows - N;
  
  return SpSubview<eT>(*this, start_row, 0, N, n_cols);
  }



template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::tail_rows(const uword N) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > n_rows), "SpMat::tail_rows(): size out of bounds");
  
  const uword start_row = n_rows - N;
  
  return SpSubview<eT>(*this, start_row, 0, N, n_cols);
  }



template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::head_cols(const uword N)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > n_cols), "SpMat::head_cols(): size out of bounds");
  
  return SpSubview<eT>(*this, 0, 0, n_rows, N);
  }



template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::head_cols(const uword N) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > n_cols), "SpMat::head_cols(): size out of bounds");
  
  return SpSubview<eT>(*this, 0, 0, n_rows, N);
  }



template<typename eT>
inline
SpSubview<eT>
SpMat<eT>::tail_cols(const uword N)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > n_cols), "SpMat::tail_cols(): size out of bounds");
  
  const uword start_col = n_cols - N;
  
  return SpSubview<eT>(*this, 0, start_col, n_rows, N);
  }



template<typename eT>
inline
const SpSubview<eT>
SpMat<eT>::tail_cols(const uword N) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (N > n_cols), "SpMat::tail_cols(): size out of bounds");
  
  const uword start_col = n_cols - N;
  
  return SpSubview<eT>(*this, 0, start_col, n_rows, N);
  }



//! creation of spdiagview (diagonal)
template<typename eT>
inline
spdiagview<eT>
SpMat<eT>::diag(const sword in_id)
  {
  arma_extra_debug_sigprint();
  
  const uword row_offset = (in_id < 0) ? uword(-in_id) : 0;
  const uword col_offset = (in_id > 0) ? uword( in_id) : 0;
  
  arma_debug_check
    (
    ((row_offset > 0) && (row_offset >= n_rows)) || ((col_offset > 0) && (col_offset >= n_cols)),
    "SpMat::diag(): requested diagonal out of bounds"
    );
  
  const uword len = (std::min)(n_rows - row_offset, n_cols - col_offset);
  
  return spdiagview<eT>(*this, row_offset, col_offset, len);
  }



//! creation of spdiagview (diagonal)
template<typename eT>
inline
const spdiagview<eT>
SpMat<eT>::diag(const sword in_id) const
  {
  arma_extra_debug_sigprint();
  
  const uword row_offset = (in_id < 0) ? -in_id : 0;
  const uword col_offset = (in_id > 0) ?  in_id : 0;
  
  arma_debug_check
    (
    ((row_offset > 0) && (row_offset >= n_rows)) || ((col_offset > 0) && (col_offset >= n_cols)),
    "SpMat::diag(): requested diagonal out of bounds"
    );
  
  const uword len = (std::min)(n_rows - row_offset, n_cols - col_offset);
  
  return spdiagview<eT>(*this, row_offset, col_offset, len);
  }



template<typename eT>
inline
void
SpMat<eT>::swap_rows(const uword in_row1, const uword in_row2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_row1 >= n_rows) || (in_row2 >= n_rows),
    "SpMat::swap_rows(): out of bounds"
    );

  // Sanity check.
  if (in_row1 == in_row2)
    {
    return;
    }

  // The easier way to do this, instead of collecting all the elements in one row and then swapping with the other, will be
  // to iterate over each column of the matrix (since we store in column-major format) and then swap the two elements in the two rows at that time.
  // We will try to avoid using the at() call since it is expensive, instead preferring to use an iterator to track our position.
  uword col1 = (in_row1 < in_row2) ? in_row1 : in_row2;
  uword col2 = (in_row1 < in_row2) ? in_row2 : in_row1;

  for (uword lcol = 0; lcol < n_cols; lcol++)
    {
    // If there is nothing in this column we can ignore it.
    if (col_ptrs[lcol] == col_ptrs[lcol + 1])
      {
      continue;
      }

    // These will represent the positions of the items themselves.
    uword loc1 = n_nonzero + 1;
    uword loc2 = n_nonzero + 1;

    for (uword search_pos = col_ptrs[lcol]; search_pos < col_ptrs[lcol + 1]; search_pos++)
      {
      if (row_indices[search_pos] == col1)
        {
        loc1 = search_pos;
        }

      if (row_indices[search_pos] == col2)
        {
        loc2 = search_pos;
        break; // No need to look any further.
        }
      }

    // There are four cases: we found both elements; we found one element (loc1); we found one element (loc2); we found zero elements.
    // If we found zero elements no work needs to be done and we can continue to the next column.
    if ((loc1 != (n_nonzero + 1)) && (loc2 != (n_nonzero + 1)))
      {
      // This is an easy case: just swap the values.  No index modifying necessary.
      eT tmp = values[loc1];
      access::rw(values[loc1]) = values[loc2];
      access::rw(values[loc2]) = tmp;
      }
    else if (loc1 != (n_nonzero + 1)) // We only found loc1 and not loc2.
      {
      // We need to find the correct place to move our value to.  It will be forward (not backwards) because in_row2 > in_row1.
      // Each iteration of the loop swaps the current value (loc1) with (loc1 + 1); in this manner we move our value down to where it should be.
      while (((loc1 + 1) < col_ptrs[lcol + 1]) && (row_indices[loc1 + 1] < in_row2))
        {
        // Swap both the values and the indices.  The column should not change.
        eT tmp = values[loc1];
        access::rw(values[loc1]) = values[loc1 + 1];
        access::rw(values[loc1 + 1]) = tmp;

        uword tmp_index = row_indices[loc1];
        access::rw(row_indices[loc1]) = row_indices[loc1 + 1];
        access::rw(row_indices[loc1 + 1]) = tmp_index;

        loc1++; // And increment the counter.
        }

      // Now set the row index correctly.
      access::rw(row_indices[loc1]) = in_row2;

      }
    else if (loc2 != (n_nonzero + 1))
      {
      // We need to find the correct place to move our value to.  It will be backwards (not forwards) because in_row1 < in_row2.
      // Each iteration of the loop swaps the current value (loc2) with (loc2 - 1); in this manner we move our value up to where it should be.
      while (((loc2 - 1) >= col_ptrs[lcol]) && (row_indices[loc2 - 1] > in_row1))
        {
        // Swap both the values and the indices.  The column should not change.
        eT tmp = values[loc2];
        access::rw(values[loc2]) = values[loc2 - 1];
        access::rw(values[loc2 - 1]) = tmp;

        uword tmp_index = row_indices[loc2];
        access::rw(row_indices[loc2]) = row_indices[loc2 - 1];
        access::rw(row_indices[loc2 - 1]) = tmp_index;

        loc2--; // And decrement the counter.
        }

      // Now set the row index correctly.
      access::rw(row_indices[loc2]) = in_row1;

      }
    /* else: no need to swap anything; both values are zero */
    }
  }



template<typename eT>
inline
void
SpMat<eT>::swap_cols(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();

  // slow but works
  for(uword lrow = 0; lrow < n_rows; ++lrow)
    {
    eT tmp = at(lrow, in_col1);
    at(lrow, in_col1) = at(lrow, in_col2);
    at(lrow, in_col2) = tmp;
    }
  }



template<typename eT>
inline
void
SpMat<eT>::shed_row(const uword row_num)
  {
  arma_extra_debug_sigprint();
  arma_debug_check (row_num >= n_rows, "SpMat::shed_row(): out of bounds");

  shed_rows (row_num, row_num);
  }



template<typename eT>
inline
void
SpMat<eT>::shed_col(const uword col_num)
  {
  arma_extra_debug_sigprint();
  arma_debug_check (col_num >= n_cols, "SpMat::shed_col(): out of bounds");

  shed_cols(col_num, col_num);
  }



template<typename eT>
inline
void
SpMat<eT>::shed_rows(const uword in_row1, const uword in_row2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "SpMat::shed_rows(): indices out of bounds or incorectly used"
    );

  uword i, j;
  // Store the length of values
  uword vlength = n_nonzero;
  // Store the length of col_ptrs
  uword clength = n_cols + 1;

  // This is O(n * n_cols) and inplace, there may be a faster way, though.
  for (i = 0, j = 0; i < vlength; ++i)
    {
    // Store the row of the ith element.
    const uword lrow = row_indices[i];
    // Is the ith element in the range of rows we want to remove?
    if (lrow >= in_row1 && lrow <= in_row2)
      {
      // Increment our "removed elements" counter.
      ++j;

      // Adjust the values of col_ptrs each time we remove an element.
      // Basically, the length of one column reduces by one, and everything to
      // its right gets reduced by one to represent all the elements being
      // shifted to the left by one.
      for(uword k = 0; k < clength; ++k)
        {
        if (col_ptrs[k] > (i - j + 1))
          {
          --access::rw(col_ptrs[k]);
          }
        }
      }
    else
      {
      // We shift the element we checked to the left by how many elements
      // we have removed.
      // j = 0 until we remove the first element.
      if (j != 0)
        {
        access::rw(row_indices[i - j]) = (lrow > in_row2) ? (lrow - (in_row2 - in_row1 + 1)) : lrow;
        access::rw(values[i - j]) = values[i];
        }
      }
    }

  // j is the number of elements removed.

  // Shrink the vectors.  This will copy the memory.
  mem_resize(n_nonzero - j);

  // Adjust row and element counts.
  access::rw(n_rows)    = n_rows - (in_row2 - in_row1) - 1;
  access::rw(n_elem)    = n_rows * n_cols;
  }



template<typename eT>
inline
void
SpMat<eT>::shed_cols(const uword in_col1, const uword in_col2)
  {
  arma_extra_debug_sigprint();

  arma_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "SpMat::shed_cols(): indices out of bounds or incorrectly used"
    );

  // First we find the locations in values and row_indices for the column entries.
  uword col_beg = col_ptrs[in_col1];
  uword col_end = col_ptrs[in_col2 + 1];

  // Then we find the number of entries in the column.
  uword diff = col_end - col_beg;

  if (diff > 0)
    {
    eT*    new_values      = memory::acquire_chunked<eT>   (n_nonzero - diff);
    uword* new_row_indices = memory::acquire_chunked<uword>(n_nonzero - diff);

    // Copy first part.
    if (col_beg != 0)
      {
      arrayops::copy(new_values, values, col_beg);
      arrayops::copy(new_row_indices, row_indices, col_beg);
      }

    // Copy second part.
    if (col_end != n_nonzero)
      {
      arrayops::copy(new_values + col_beg, values + col_end, n_nonzero - col_end);
      arrayops::copy(new_row_indices + col_beg, row_indices + col_end, n_nonzero - col_end);
      }

    memory::release(values);
    memory::release(row_indices);

    access::rw(values)      = new_values;
    access::rw(row_indices) = new_row_indices;

    // Update counts and such.
    access::rw(n_nonzero) -= diff;
    }
  
  // Update column pointers.
  const uword new_n_cols = n_cols - ((in_col2 - in_col1) + 1);
  
  uword* new_col_ptrs = memory::acquire<uword>(new_n_cols + 2);
  new_col_ptrs[new_n_cols + 1] = std::numeric_limits<uword>::max();
  
  // Copy first set of columns (no manipulation required).
  if (in_col1 != 0)
    {
    arrayops::copy(new_col_ptrs, col_ptrs, in_col1);
    }
  
  // Copy second set of columns (manipulation required).
  uword cur_col = in_col1;
  for (uword i = in_col2 + 1; i <= n_cols; ++i, ++cur_col)
    {
    new_col_ptrs[cur_col] = col_ptrs[i] - diff;
    }
  
  memory::release(col_ptrs);
  access::rw(col_ptrs) = new_col_ptrs;
  
  // We update the element and column counts, and we're done.
  access::rw(n_cols) = new_n_cols;
  access::rw(n_elem) = n_cols * n_rows;
  }



/**
 * Element access; acces the i'th element (works identically to the Mat accessors).
 * If there is nothing at element i, 0 is returned.
 *
 * @param i Element to access.
 */

template<typename eT>
arma_inline
arma_warn_unused
SpValProxy<SpMat<eT> >
SpMat<eT>::operator[](const uword i)
  {
  return get_value(i);
  }



template<typename eT>
arma_inline
arma_warn_unused
eT
SpMat<eT>::operator[](const uword i) const
  {
  return get_value(i);
  }



template<typename eT>
arma_inline
arma_warn_unused
SpValProxy<SpMat<eT> >
SpMat<eT>::at(const uword i)
  {
  return get_value(i);
  }



template<typename eT>
arma_inline
arma_warn_unused
eT
SpMat<eT>::at(const uword i) const
  {
  return get_value(i);
  }



template<typename eT>
arma_inline
arma_warn_unused
SpValProxy<SpMat<eT> >
SpMat<eT>::operator()(const uword i)
  {
  arma_debug_check( (i >= n_elem), "SpMat::operator(): out of bounds");
  return get_value(i);
  }



template<typename eT>
arma_inline
arma_warn_unused
eT
SpMat<eT>::operator()(const uword i) const
  {
  arma_debug_check( (i >= n_elem), "SpMat::operator(): out of bounds");
  return get_value(i);
  }



/**
 * Element access; access the element at row in_rows and column in_col.
 * If there is nothing at that position, 0 is returned.
 */

template<typename eT>
arma_inline
arma_warn_unused
SpValProxy<SpMat<eT> >
SpMat<eT>::at(const uword in_row, const uword in_col)
  {
  return get_value(in_row, in_col);
  }



template<typename eT>
arma_inline
arma_warn_unused
eT
SpMat<eT>::at(const uword in_row, const uword in_col) const
  {
  return get_value(in_row, in_col);
  }



template<typename eT>
arma_inline
arma_warn_unused
SpValProxy<SpMat<eT> >
SpMat<eT>::operator()(const uword in_row, const uword in_col)
  {
  arma_debug_check( ((in_row >= n_rows) || (in_col >= n_cols)), "SpMat::operator(): out of bounds");
  return get_value(in_row, in_col);
  }



template<typename eT>
arma_inline
arma_warn_unused
eT
SpMat<eT>::operator()(const uword in_row, const uword in_col) const
  {
  arma_debug_check( ((in_row >= n_rows) || (in_col >= n_cols)), "SpMat::operator(): out of bounds");
  return get_value(in_row, in_col);
  }



/**
 * Check if matrix is empty (no size, no values).
 */
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::is_empty() const
  {
  return(n_elem == 0);
  }



//! returns true if the object can be interpreted as a column or row vector
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::is_vec() const
  {
  return ( (n_rows == 1) || (n_cols == 1) );
  }



//! returns true if the object can be interpreted as a row vector
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::is_rowvec() const
  {
  return (n_rows == 1);
  }



//! returns true if the object can be interpreted as a column vector
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::is_colvec() const
  {
  return (n_cols == 1);
  }



//! returns true if the object has the same number of non-zero rows and columnns
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::is_square() const
  {
  return (n_rows == n_cols);
  }



//! returns true if all of the elements are finite
template<typename eT>
inline
arma_warn_unused
bool
SpMat<eT>::is_finite() const
  {
  return arrayops::is_finite(values, n_nonzero);
  }



//! returns true if the given index is currently in range
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const uword i) const
  {
  return (i < n_elem);
  }


//! returns true if the given start and end indices are currently in range
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const span& x) const
  {
  arma_extra_debug_sigprint();

  if(x.whole == true)
    {
    return true;
    }
  else
    {
    const uword a = x.a;
    const uword b = x.b;

    return ( (a <= b) && (b < n_elem) );
    }
  }



//! returns true if the given location is currently in range
template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const uword in_row, const uword in_col) const
  {
  return ( (in_row < n_rows) && (in_col < n_cols) );
  }



template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const span& row_span, const uword in_col) const
  {
  arma_extra_debug_sigprint();

  if(row_span.whole == true)
    {
    return (in_col < n_cols);
    }
  else
    {
    const uword in_row1 = row_span.a;
    const uword in_row2 = row_span.b;

    return ( (in_row1 <= in_row2) && (in_row2 < n_rows) && (in_col < n_cols) );
    }
  }



template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const uword in_row, const span& col_span) const
  {
  arma_extra_debug_sigprint();

  if(col_span.whole == true)
    {
    return (in_row < n_rows);
    }
  else
    {
    const uword in_col1 = col_span.a;
    const uword in_col2 = col_span.b;

    return ( (in_row < n_rows) && (in_col1 <= in_col2) && (in_col2 < n_cols) );
    }
  }



template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const span& row_span, const span& col_span) const
  {
  arma_extra_debug_sigprint();

  const uword in_row1 = row_span.a;
  const uword in_row2 = row_span.b;

  const uword in_col1 = col_span.a;
  const uword in_col2 = col_span.b;

  const bool rows_ok = row_span.whole ? true : ( (in_row1 <= in_row2) && (in_row2 < n_rows) );
  const bool cols_ok = col_span.whole ? true : ( (in_col1 <= in_col2) && (in_col2 < n_cols) );

  return ( (rows_ok == true) && (cols_ok == true) );
  }



template<typename eT>
arma_inline
arma_warn_unused
bool
SpMat<eT>::in_range(const uword in_row, const uword in_col, const SizeMat& s) const
  {
  const uword l_n_rows = n_rows;
  const uword l_n_cols = n_cols;
  
  if( (in_row >= l_n_rows) || (in_col >= l_n_cols) || ((in_row + s.n_rows) > l_n_rows) || ((in_col + s.n_cols) > l_n_cols) )
    {
    return false;
    }
  else
    {
    return true;
    }
  }



template<typename eT>
inline
void
SpMat<eT>::impl_print(const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = ARMA_DEFAULT_OSTREAM.width();

    ARMA_DEFAULT_OSTREAM << extra_text << '\n';

    ARMA_DEFAULT_OSTREAM.width(orig_width);
    }

  arma_ostream::print(ARMA_DEFAULT_OSTREAM, *this, true);
  }



template<typename eT>
inline
void
SpMat<eT>::impl_print(std::ostream& user_stream, const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = user_stream.width();

    user_stream << extra_text << '\n';

    user_stream.width(orig_width);
    }

  arma_ostream::print(user_stream, *this, true);
  }



template<typename eT>
inline
void
SpMat<eT>::impl_raw_print(const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = ARMA_DEFAULT_OSTREAM.width();

    ARMA_DEFAULT_OSTREAM << extra_text << '\n';

    ARMA_DEFAULT_OSTREAM.width(orig_width);
    }

  arma_ostream::print(ARMA_DEFAULT_OSTREAM, *this, false);
  }


template<typename eT>
inline
void
SpMat<eT>::impl_raw_print(std::ostream& user_stream, const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = user_stream.width();

    user_stream << extra_text << '\n';

    user_stream.width(orig_width);
    }

  arma_ostream::print(user_stream, *this, false);
  }



/**
 * Matrix printing, prepends supplied text.
 * Prints 0 wherever no element exists.
 */
template<typename eT>
inline
void
SpMat<eT>::impl_print_dense(const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = ARMA_DEFAULT_OSTREAM.width();

    ARMA_DEFAULT_OSTREAM << extra_text << '\n';

    ARMA_DEFAULT_OSTREAM.width(orig_width);
    }

  arma_ostream::print_dense(ARMA_DEFAULT_OSTREAM, *this, true);
  }



template<typename eT>
inline
void
SpMat<eT>::impl_print_dense(std::ostream& user_stream, const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = user_stream.width();

    user_stream << extra_text << '\n';

    user_stream.width(orig_width);
    }

  arma_ostream::print_dense(user_stream, *this, true);
  }



template<typename eT>
inline
void
SpMat<eT>::impl_raw_print_dense(const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = ARMA_DEFAULT_OSTREAM.width();

    ARMA_DEFAULT_OSTREAM << extra_text << '\n';

    ARMA_DEFAULT_OSTREAM.width(orig_width);
    }

  arma_ostream::print_dense(ARMA_DEFAULT_OSTREAM, *this, false);
  }



template<typename eT>
inline
void
SpMat<eT>::impl_raw_print_dense(std::ostream& user_stream, const std::string& extra_text) const
  {
  arma_extra_debug_sigprint();

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = user_stream.width();

    user_stream << extra_text << '\n';

    user_stream.width(orig_width);
    }

  arma_ostream::print_dense(user_stream, *this, false);
  }



//! Set the size to the size of another matrix.
template<typename eT>
template<typename eT2>
inline
void
SpMat<eT>::copy_size(const SpMat<eT2>& m)
  {
  arma_extra_debug_sigprint();

  init(m.n_rows, m.n_cols);
  }



template<typename eT>
template<typename eT2>
inline
void
SpMat<eT>::copy_size(const Mat<eT2>& m)
  {
  arma_extra_debug_sigprint();

  init(m.n_rows, m.n_cols);
  }



/**
 * Set the size of the matrix; the matrix will be sized as a column vector
 *
 * @param in_elem Number of elements to allow.
 */
template<typename eT>
inline
void
SpMat<eT>::set_size(const uword in_elem)
  {
  arma_extra_debug_sigprint();

  // If this is a row vector, we resize to a row vector.
  if(vec_state == 2)
    {
    init(1, in_elem);
    }
  else
    {
    init(in_elem, 1);
    }
  }



/**
 * Set the size of the matrix
 *
 * @param in_rows Number of rows to allow.
 * @param in_cols Number of columns to allow.
 */
template<typename eT>
inline
void
SpMat<eT>::set_size(const uword in_rows, const uword in_cols)
  {
  arma_extra_debug_sigprint();

  init(in_rows, in_cols);
  }



template<typename eT>
inline
void
SpMat<eT>::reshape(const uword in_rows, const uword in_cols, const uword dim)
  {
  arma_extra_debug_sigprint();

  if (dim == 0)
    {
    // We have to modify all of the relevant row indices and the relevant column pointers.
    // Iterate over all the points to do this.  We won't be deleting any points, but we will be modifying
    // columns and rows. We'll have to store a new set of column vectors.
    uword* new_col_ptrs    = memory::acquire<uword>(in_cols + 2);
    new_col_ptrs[in_cols + 1] = std::numeric_limits<uword>::max();
    
    uword* new_row_indices = memory::acquire_chunked<uword>(n_nonzero + 1);
    access::rw(new_row_indices[n_nonzero]) = 0;

    arrayops::inplace_set(new_col_ptrs, uword(0), in_cols + 1);

    for(const_iterator it = begin(); it != end(); it++)
      {
      uword vector_position = (it.col() * n_rows) + it.row();
      new_row_indices[it.pos()] = vector_position % in_rows;
      ++new_col_ptrs[vector_position / in_rows + 1];
      }

    // Now sum the column counts to get the new column pointers.
    for(uword i = 1; i <= in_cols; i++)
      {
      access::rw(new_col_ptrs[i]) += new_col_ptrs[i - 1];
      }

    // Copy the new row indices.
    memory::release(row_indices);
    access::rw(row_indices) = new_row_indices;

    memory::release(col_ptrs);
    access::rw(col_ptrs) = new_col_ptrs;

    // Now set the size.
    access::rw(n_rows) = in_rows;
    access::rw(n_cols) = in_cols;
    }
  else
    {
    // Row-wise reshaping.  This is more tedious and we will use a separate sparse matrix to do it.
    SpMat<eT> tmp(in_rows, in_cols);

    for(const_row_iterator it = begin_row(); it.pos() < n_nonzero; it++)
      {
      uword vector_position = (it.row() * n_cols) + it.col();

      tmp((vector_position / in_cols), (vector_position % in_cols)) = (*it);
      }

    (*this).operator=(tmp);
    }
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::zeros()
  {
  arma_extra_debug_sigprint();

  if (n_nonzero > 0)
    {
    memory::release(values);
    memory::release(row_indices);

    access::rw(values)      = memory::acquire_chunked<eT>(1);
    access::rw(row_indices) = memory::acquire_chunked<uword>(1);

    access::rw(values[0]) = 0;
    access::rw(row_indices[0]) = 0;
    }

  access::rw(n_nonzero) = 0;
  arrayops::inplace_set(access::rwp(col_ptrs), uword(0), n_cols + 1);

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::zeros(const uword in_elem)
  {
  arma_extra_debug_sigprint();

  if(vec_state == 2)
    {
    init(1, in_elem); // Row vector
    }
  else
    {
    init(in_elem, 1);
    }

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::zeros(const uword in_rows, const uword in_cols)
  {
  arma_extra_debug_sigprint();

  init(in_rows, in_cols);

  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::eye()
  {
  arma_extra_debug_sigprint();

  return (*this).eye(n_rows, n_cols);
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::eye(const uword in_rows, const uword in_cols)
  {
  arma_extra_debug_sigprint();
  
  const uword N = (std::min)(in_rows, in_cols);
  
  init(in_rows, in_cols);
  
  mem_resize(N);
  
  arrayops::inplace_set(access::rwp(values), eT(1), N);
  
  for(uword i = 0; i <  N; ++i) { access::rw(row_indices[i]) = i; }
  
  for(uword i = 0; i <= N; ++i) { access::rw(col_ptrs[i])    = i; }
  
  access::rw(n_nonzero) = N;
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::speye()
  {
  arma_extra_debug_sigprint();

  return (*this).eye(n_rows, n_cols);
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::speye(const uword in_n_rows, const uword in_n_cols)
  {
  arma_extra_debug_sigprint();
  
  return (*this).eye(in_n_rows, in_n_cols);
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::sprandu(const uword in_rows, const uword in_cols, const double density)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (density < double(0)) || (density > double(1)) ), "sprandu(): density must be in the [0,1] interval" );
  
  zeros(in_rows, in_cols);
  
  mem_resize( uword(density * double(in_rows) * double(in_cols) + 0.5) );
  
  if(n_nonzero == 0)
    {
    return *this;
    }
  
  arma_rng::randu<eT>::fill( access::rwp(values), n_nonzero );
  
  uvec indices = linspace<uvec>( 0u, in_rows*in_cols-1, n_nonzero );
  
  // perturb the indices
  for(uword i=1; i < n_nonzero-1; ++i)
    {
    const uword index_left  = indices[i-1];
    const uword index_right = indices[i+1];
    
    const uword center = (index_left + index_right) / 2;
    
    const uword delta1 = center      - index_left - 1;
    const uword delta2 = index_right - center     - 1;
    
    const uword min_delta = (std::min)(delta1, delta2);
    
    uword index_new = uword( double(center) + double(min_delta) * (2.0*randu()-1.0) );
    
    // paranoia, but better be safe than sorry
    if( (index_left < index_new) && (index_new < index_right) )
      {
      indices[i] = index_new;
      }
    }
  
  uword cur_index = 0;
  uword count     = 0;  
  
  for(uword lcol = 0; lcol < in_cols; ++lcol)
  for(uword lrow = 0; lrow < in_rows; ++lrow)
    {
    if(count == indices[cur_index])
      {
      access::rw(row_indices[cur_index]) = lrow;
      access::rw(col_ptrs[lcol + 1])++;
      ++cur_index;
      }
    
    ++count;
    }
  
  if(cur_index != n_nonzero)
    {
    // Fix size to correct size.
    mem_resize(cur_index);
    }
  
  // Sum column pointers.
  for(uword lcol = 1; lcol <= in_cols; ++lcol)
    {
    access::rw(col_ptrs[lcol]) += col_ptrs[lcol - 1];
    }
  
  return *this;
  }



template<typename eT>
inline
const SpMat<eT>&
SpMat<eT>::sprandn(const uword in_rows, const uword in_cols, const double density)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( ( (density < double(0)) || (density > double(1)) ), "sprandn(): density must be in the [0,1] interval" );
  
  zeros(in_rows, in_cols);
  
  mem_resize( uword(density * double(in_rows) * double(in_cols) + 0.5) );
  
  if(n_nonzero == 0)
    {
    return *this;
    }
  
  arma_rng::randn<eT>::fill( access::rwp(values), n_nonzero );
  
  uvec indices = linspace<uvec>( 0u, in_rows*in_cols-1, n_nonzero );
  
  // perturb the indices
  for(uword i=1; i < n_nonzero-1; ++i)
    {
    const uword index_left  = indices[i-1];
    const uword index_right = indices[i+1];
    
    const uword center = (index_left + index_right) / 2;
    
    const uword delta1 = center      - index_left - 1;
    const uword delta2 = index_right - center     - 1;
    
    const uword min_delta = (std::min)(delta1, delta2);
    
    uword index_new = uword( double(center) + double(min_delta) * (2.0*randu()-1.0) );
    
    // paranoia, but better be safe than sorry
    if( (index_left < index_new) && (index_new < index_right) )
      {
      indices[i] = index_new;
      }
    }
  
  uword cur_index = 0;
  uword count     = 0;  
  
  for(uword lcol = 0; lcol < in_cols; ++lcol)
  for(uword lrow = 0; lrow < in_rows; ++lrow)
    {
    if(count == indices[cur_index])
      {
      access::rw(row_indices[cur_index]) = lrow;
      access::rw(col_ptrs[lcol + 1])++;
      ++cur_index;
      }
    
    ++count;
    }
  
  if(cur_index != n_nonzero)
    {
    // Fix size to correct size.
    mem_resize(cur_index);
    }
  
  // Sum column pointers.
  for(uword lcol = 1; lcol <= in_cols; ++lcol)
    {
    access::rw(col_ptrs[lcol]) += col_ptrs[lcol - 1];
    }
  
  return *this;
  }



template<typename eT>
inline
void
SpMat<eT>::reset()
  {
  arma_extra_debug_sigprint();

  switch(vec_state)
    {
    default:
      init(0, 0);
      break;
      
    case 1:
      init(0, 1);
      break;
    
    case 2:
      init(1, 0);
      break;
    }
  }



//! save the matrix to a file
template<typename eT>
inline
bool
SpMat<eT>::save(const std::string name, const file_type type, const bool print_status) const
  {
  arma_extra_debug_sigprint();
  
  bool save_okay;
  
  switch(type)
    {
    // case raw_ascii:
    //   save_okay = diskio::save_raw_ascii(*this, name);
    //   break;
    
    // case csv_ascii:
    //   save_okay = diskio::save_csv_ascii(*this, name);
    //   break;
    
    case arma_binary:
      save_okay = diskio::save_arma_binary(*this, name);
      break;
    
    case coord_ascii:
      save_okay = diskio::save_coord_ascii(*this, name);
      break;
    
    default:
      arma_warn(print_status, "SpMat::save(): unsupported file type");
      save_okay = false;
    }
  
  arma_warn( print_status && (save_okay == false), "SpMat::save(): couldn't write to ", name);
  
  return save_okay;
  }



//! save the matrix to a stream
template<typename eT>
inline
bool
SpMat<eT>::save(std::ostream& os, const file_type type, const bool print_status) const
  {
  arma_extra_debug_sigprint();
  
  bool save_okay;
  
  switch(type)
    {
    // case raw_ascii:
    //   save_okay = diskio::save_raw_ascii(*this, os);
    //   break;
    
    // case csv_ascii:
    //   save_okay = diskio::save_csv_ascii(*this, os);
    //   break;
    
    case arma_binary:
      save_okay = diskio::save_arma_binary(*this, os);
      break;
    
    case coord_ascii:
      save_okay = diskio::save_coord_ascii(*this, os);
      break;
    
    default:
      arma_warn(print_status, "SpMat::save(): unsupported file type");
      save_okay = false;
    }
  
  arma_warn( print_status && (save_okay == false), "SpMat::save(): couldn't write to the given stream");
  
  return save_okay;
  }



//! load a matrix from a file
template<typename eT>
inline
bool
SpMat<eT>::load(const std::string name, const file_type type, const bool print_status)
  {
  arma_extra_debug_sigprint();
  
  bool load_okay;
  std::string err_msg;
  
  switch(type)
    {
    // case auto_detect:
    //   load_okay = diskio::load_auto_detect(*this, name, err_msg);
    //   break;
    
    // case raw_ascii:
    //   load_okay = diskio::load_raw_ascii(*this, name, err_msg);
    //   break;
    
    // case csv_ascii:
    //   load_okay = diskio::load_csv_ascii(*this, name, err_msg);
    //   break;
    
    case arma_binary:
      load_okay = diskio::load_arma_binary(*this, name, err_msg);
      break;
    
    case coord_ascii:
      load_okay = diskio::load_coord_ascii(*this, name, err_msg);
      break;
    
    default:
      arma_warn(print_status, "SpMat::load(): unsupported file type");
      load_okay = false;
    }
  
  if(load_okay == false)
    {
    if(err_msg.length() > 0)
      {
      arma_warn(print_status, "SpMat::load(): ", err_msg, name);
      }
    else
      {
      arma_warn(print_status, "SpMat::load(): couldn't read ", name);
      }
    }
  
  if(load_okay == false)
    {
    (*this).reset();
    }
    
  return load_okay;
  }



//! load a matrix from a stream
template<typename eT>
inline
bool
SpMat<eT>::load(std::istream& is, const file_type type, const bool print_status)
  {
  arma_extra_debug_sigprint();
  
  bool load_okay;
  std::string err_msg;
  
  switch(type)
    {
    // case auto_detect:
    //   load_okay = diskio::load_auto_detect(*this, is, err_msg);
    //   break;
    
    // case raw_ascii:
    //   load_okay = diskio::load_raw_ascii(*this, is, err_msg);
    //   break;
    
    // case csv_ascii:
    //   load_okay = diskio::load_csv_ascii(*this, is, err_msg);
    //   break;
    
    case arma_binary:
      load_okay = diskio::load_arma_binary(*this, is, err_msg);
      break;
    
    case coord_ascii:
      load_okay = diskio::load_coord_ascii(*this, is, err_msg);
      break;
    
    default:
      arma_warn(print_status, "SpMat::load(): unsupported file type");
      load_okay = false;
    }
  
  
  if(load_okay == false)
    {
    if(err_msg.length() > 0)
      {
      arma_warn(print_status, "SpMat::load(): ", err_msg, "the given stream");
      }
    else
      {
      arma_warn(print_status, "SpMat::load(): couldn't load from the given stream");
      }
    }
  
  if(load_okay == false)
    {
    (*this).reset();
    }
    
  return load_okay;
  }



//! save the matrix to a file, without printing any error messages
template<typename eT>
inline
bool
SpMat<eT>::quiet_save(const std::string name, const file_type type) const
  {
  arma_extra_debug_sigprint();
  
  return (*this).save(name, type, false);
  }



//! save the matrix to a stream, without printing any error messages
template<typename eT>
inline
bool
SpMat<eT>::quiet_save(std::ostream& os, const file_type type) const
  {
  arma_extra_debug_sigprint();
  
  return (*this).save(os, type, false);
  }



//! load a matrix from a file, without printing any error messages
template<typename eT>
inline
bool
SpMat<eT>::quiet_load(const std::string name, const file_type type)
  {
  arma_extra_debug_sigprint();
  
  return (*this).load(name, type, false);
  }



//! load a matrix from a stream, without printing any error messages
template<typename eT>
inline
bool
SpMat<eT>::quiet_load(std::istream& is, const file_type type)
  {
  arma_extra_debug_sigprint();
  
  return (*this).load(is, type, false);
  }



/**
 * Initialize the matrix to the specified size.  Data is not preserved, so the matrix is assumed to be entirely sparse (empty).
 */
template<typename eT>
inline
void
SpMat<eT>::init(uword in_rows, uword in_cols)
  {
  arma_extra_debug_sigprint();
  
  // Verify that we are allowed to do this.
  if(vec_state > 0)
    {
    if((in_rows == 0) && (in_cols == 0))
      {
      if(vec_state == 1)
        {
        in_cols = 1;
        }
      else
      if(vec_state == 2)
        {
        in_rows = 1;
        }
      }
    else
      {
      arma_debug_check
        (
        ( ((vec_state == 1) && (in_cols != 1)) || ((vec_state == 2) && (in_rows != 1)) ),
        "SpMat::init(): object is a row or column vector; requested size is not compatible"
        );
      }
    }
  
  #if (defined(ARMA_USE_CXX11) || defined(ARMA_64BIT_WORD))
    const char* error_message = "SpMat::init(): requested size is too large";
  #else
    const char* error_message = "SpMat::init(): requested size is too large; suggest to compile in C++11 mode or enable ARMA_64BIT_WORD";
  #endif
  
  // Ensure that n_elem can hold the result of (n_rows * n_cols)
  arma_debug_check
    (
      (
      ( (in_rows > ARMA_MAX_UHWORD) || (in_cols > ARMA_MAX_UHWORD) )
        ? ( (double(in_rows) * double(in_cols)) > double(ARMA_MAX_UWORD) )
        : false
      ),
      error_message
    );
  
  // Clean out the existing memory.
  if (values)
    {
    memory::release(values);
    memory::release(row_indices);
    }
  
  access::rw(values)      = memory::acquire_chunked<eT>   (1);
  access::rw(row_indices) = memory::acquire_chunked<uword>(1);
  
  access::rw(values[0])      = 0;
  access::rw(row_indices[0]) = 0;
  
  memory::release(col_ptrs);
  
  // Set the new size accordingly.
  access::rw(n_rows)    = in_rows;
  access::rw(n_cols)    = in_cols;
  access::rw(n_elem)    = (in_rows * in_cols);
  access::rw(n_nonzero) = 0;
  
  // Try to allocate the column pointers, filling them with 0,
  // except for the last element which contains the maximum possible element
  // (so iterators terminate correctly).
  access::rw(col_ptrs) = memory::acquire<uword>(in_cols + 2);
  
  arrayops::inplace_set(access::rwp(col_ptrs), uword(0), in_cols + 1);
  
  access::rw(col_ptrs[in_cols + 1]) = std::numeric_limits<uword>::max();
  }



/**
 * Initialize the matrix from a string.
 */
template<typename eT>
inline
void
SpMat<eT>::init(const std::string& text)
  {
  arma_extra_debug_sigprint();

  // Figure out the size first.
  uword t_n_rows = 0;
  uword t_n_cols = 0;

  bool t_n_cols_found = false;

  std::string token;

  std::string::size_type line_start = 0;
  std::string::size_type   line_end = 0;

  while (line_start < text.length())
    {

    line_end = text.find(';', line_start);

    if (line_end == std::string::npos)
      line_end = text.length() - 1;

    std::string::size_type line_len = line_end - line_start + 1;
    std::stringstream line_stream(text.substr(line_start, line_len));

    // Step through each column.
    uword line_n_cols = 0;

    while (line_stream >> token)
      {
      ++line_n_cols;
      }

    if (line_n_cols > 0)
      {
      if (t_n_cols_found == false)
        {
        t_n_cols = line_n_cols;
        t_n_cols_found = true;
        }
      else // Check it each time through, just to make sure.
        arma_check((line_n_cols != t_n_cols), "SpMat::init(): inconsistent number of columns in given string");

      ++t_n_rows;
      }

    line_start = line_end + 1;

    }

  set_size(t_n_rows, t_n_cols);

  // Second time through will pick up all the values.
  line_start = 0;
  line_end = 0;

  uword lrow = 0;

  while (line_start < text.length())
    {

    line_end = text.find(';', line_start);

    if (line_end == std::string::npos)
      line_end = text.length() - 1;

    std::string::size_type line_len = line_end - line_start + 1;
    std::stringstream line_stream(text.substr(line_start, line_len));

    uword lcol = 0;
    eT val;

    while (line_stream >> val)
      {
      // Only add nonzero elements.
      if (val != eT(0))
        {
        get_value(lrow, lcol) = val;
        }

      ++lcol;
      }

    ++lrow;
    line_start = line_end + 1;

    }

  }

/**
 * Copy from another matrix.
 */
template<typename eT>
inline
void
SpMat<eT>::init(const SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  // Ensure we are not initializing to ourselves.
  if (this != &x)
    {
    init(x.n_rows, x.n_cols);

    // values and row_indices may not be null.
    if (values != NULL)
      {
      memory::release(values);
      memory::release(row_indices);
      }

    access::rw(values)      = memory::acquire_chunked<eT>   (x.n_nonzero + 1);
    access::rw(row_indices) = memory::acquire_chunked<uword>(x.n_nonzero + 1);

    // Now copy over the elements.
    arrayops::copy(access::rwp(values),      x.values,      x.n_nonzero + 1);
    arrayops::copy(access::rwp(row_indices), x.row_indices, x.n_nonzero + 1);
    arrayops::copy(access::rwp(col_ptrs),    x.col_ptrs,    x.n_cols + 1);
    
    access::rw(n_nonzero) = x.n_nonzero;
    }
  }



template<typename eT>
inline
void
SpMat<eT>::init_batch_std(const Mat<uword>& locs, const Mat<eT>& vals, const bool sort_locations)
  {
  arma_extra_debug_sigprint();
  
  // Resize to correct number of elements.
  mem_resize(vals.n_elem);
  
  // Reset column pointers to zero.
  arrayops::inplace_set(access::rwp(col_ptrs), uword(0), n_cols + 1);
  
  bool actually_sorted = true;
  
  if(sort_locations == true)
    {
    // sort_index() uses std::sort() which may use quicksort... so we better
    // make sure it's not already sorted before taking an O(N^2) sort penalty.
    for (uword i = 1; i < locs.n_cols; ++i)
      {
      const uword* locs_i   = locs.colptr(i  );
      const uword* locs_im1 = locs.colptr(i-1);
      
      if( (locs_i[1] < locs_im1[1]) || (locs_i[1] == locs_im1[1]  &&  locs_i[0] <= locs_im1[0]) )
        {
        actually_sorted = false;
        break;
        }
      }
    
    if(actually_sorted == false)
      {
      // This may not be the fastest possible implementation but it maximizes code reuse.
      Col<uword> abslocs(locs.n_cols);
      
      for (uword i = 0; i < locs.n_cols; ++i)
        {
        const uword* locs_i = locs.colptr(i);
        
        abslocs[i] = locs_i[1] * n_rows + locs_i[0];
        }
      
      uvec sorted_indices = sort_index(abslocs); // Ascending sort.
      
      // Now we add the elements in this sorted order.
      for (uword i = 0; i < sorted_indices.n_elem; ++i)
        {
        const uword* locs_i = locs.colptr( sorted_indices[i] );
        
        arma_debug_check( ( (locs_i[0] >= n_rows) || (locs_i[1] >= n_cols) ), "SpMat::SpMat(): invalid row or column index" );
        
        if(i > 0)
          {
          const uword* locs_im1 = locs.colptr( sorted_indices[i-1] );
          
          arma_debug_check( ( (locs_i[1] == locs_im1[1]) && (locs_i[0] == locs_im1[0]) ), "SpMat::SpMat(): detected identical locations" );
          }
        
        access::rw(values[i])      = vals[ sorted_indices[i] ];
        access::rw(row_indices[i]) = locs_i[0];
        
        access::rw(col_ptrs[ locs_i[1] + 1 ])++;
        }
      }
    }
  
  if( (sort_locations == false) || (actually_sorted == true) )
    {
    // Now set the values and row indices correctly.
    // Increment the column pointers in each column (so they are column "counts").
    for(uword i = 0; i < vals.n_elem; ++i)
      {
      const uword* locs_i = locs.colptr(i);
      
      arma_debug_check( ( (locs_i[0] >= n_rows) || (locs_i[1] >= n_cols) ), "SpMat::SpMat(): invalid row or column index" );
      
      if(i > 0)
        {
        const uword* locs_im1 = locs.colptr(i-1);
        
        arma_debug_check
          (
          ( (locs_i[1] < locs_im1[1]) || (locs_i[1] == locs_im1[1]  &&  locs_i[0] < locs_im1[0]) ),
          "SpMat::SpMat(): out of order points; either pass sort_locations = true, or sort points in column-major ordering"
          );
        
        arma_debug_check( ( (locs_i[1] == locs_im1[1]) && (locs_i[0] == locs_im1[0]) ), "SpMat::SpMat(): detected identical locations" );
        }
      
      access::rw(values[i])      = vals[i];
      access::rw(row_indices[i]) = locs_i[0];
      
      access::rw(col_ptrs[ locs_i[1] + 1 ])++;
      }
    }
  
  // Now fix the column pointers.
  for (uword i = 0; i < n_cols; ++i)
    {
    access::rw(col_ptrs[i + 1]) += col_ptrs[i];
    }
  }



template<typename eT>
inline
void
SpMat<eT>::init_batch_add(const Mat<uword>& locs, const Mat<eT>& vals, const bool sort_locations)
  {
  arma_extra_debug_sigprint();
  
  if(locs.n_cols < 2)
    {
    init_batch_std(locs, vals, false);
    return;
    }
  
  // Reset column pointers to zero.
  arrayops::inplace_set(access::rwp(col_ptrs), uword(0), n_cols + 1);
  
  bool actually_sorted = true;
  
  if(sort_locations == true)
    {
    // sort_index() uses std::sort() which may use quicksort... so we better
    // make sure it's not already sorted before taking an O(N^2) sort penalty.
    for (uword i = 1; i < locs.n_cols; ++i)
      {
      const uword* locs_i   = locs.colptr(i  );
      const uword* locs_im1 = locs.colptr(i-1);
      
      if( (locs_i[1] < locs_im1[1]) || (locs_i[1] == locs_im1[1]  &&  locs_i[0] <= locs_im1[0]) )
        {
        actually_sorted = false;
        break;
        }
      }
    
    if(actually_sorted == false)
      {
      // This may not be the fastest possible implementation but it maximizes code reuse.
      Col<uword> abslocs(locs.n_cols);
      
      for (uword i = 0; i < locs.n_cols; ++i)
        {
        const uword* locs_i = locs.colptr(i);
        
        abslocs[i] = locs_i[1] * n_rows + locs_i[0];
        }
      
      uvec sorted_indices = sort_index(abslocs); // Ascending sort.
      
      // work out the number of unique elments 
      uword n_unique = 1;  // first element is unique
      
      for(uword i=1; i < sorted_indices.n_elem; ++i)
        {
        const uword* locs_i   = locs.colptr( sorted_indices[i  ] );
        const uword* locs_im1 = locs.colptr( sorted_indices[i-1] );
        
        if( (locs_i[1] != locs_im1[1]) || (locs_i[0] != locs_im1[0]) )  { ++n_unique; }
        }
      
      // resize to correct number of elements
      mem_resize(n_unique);
      
      // Now we add the elements in this sorted order.
      uword count = 0;
      
      // first element
        {
        const uword  i      = 0;
        const uword* locs_i = locs.colptr( sorted_indices[i] );
        
        arma_debug_check( ( (locs_i[0] >= n_rows) || (locs_i[1] >= n_cols) ), "SpMat::SpMat(): invalid row or column index" );
        
        access::rw(values[count])      = vals[ sorted_indices[i] ];
        access::rw(row_indices[count]) = locs_i[0];
        
        access::rw(col_ptrs[ locs_i[1] + 1 ])++;
        }
      
      for(uword i=1; i < sorted_indices.n_elem; ++i)
        {
        const uword* locs_i   = locs.colptr( sorted_indices[i  ] );
        const uword* locs_im1 = locs.colptr( sorted_indices[i-1] );
        
        arma_debug_check( ( (locs_i[0] >= n_rows) || (locs_i[1] >= n_cols) ), "SpMat::SpMat(): invalid row or column index" );
        
        if( (locs_i[1] == locs_im1[1]) && (locs_i[0] == locs_im1[0]) )
          {
          access::rw(values[count]) += vals[ sorted_indices[i] ];
          }
        else
          {
          count++;
          access::rw(values[count])      = vals[ sorted_indices[i] ];
          access::rw(row_indices[count]) = locs_i[0];
          
          access::rw(col_ptrs[ locs_i[1] + 1 ])++;
          }
        }
      }
    }
  
  if( (sort_locations == false) || (actually_sorted == true) )
    {
    // work out the number of unique elments 
    uword n_unique = 1;  // first element is unique
    
    for(uword i=1; i < locs.n_cols; ++i)
      {
      const uword* locs_i   = locs.colptr(i  );
      const uword* locs_im1 = locs.colptr(i-1);
      
      if( (locs_i[1] != locs_im1[1]) || (locs_i[0] != locs_im1[0]) )  { ++n_unique; }
      }
    
    // resize to correct number of elements
    mem_resize(n_unique);
    
    // Now set the values and row indices correctly.
    // Increment the column pointers in each column (so they are column "counts").
    
    uword count = 0;
    
    // first element
      {
      const uword  i      = 0;
      const uword* locs_i = locs.colptr(i);
      
      arma_debug_check( ( (locs_i[0] >= n_rows) || (locs_i[1] >= n_cols) ), "SpMat::SpMat(): invalid row or column index" );
      
      access::rw(values[count])      = vals[i];
      access::rw(row_indices[count]) = locs_i[0];
      
      access::rw(col_ptrs[ locs_i[1] + 1 ])++;
      }
    
    for(uword i=1; i < locs.n_cols; ++i)
      {
      const uword* locs_i   = locs.colptr(i  );
      const uword* locs_im1 = locs.colptr(i-1);
      
      arma_debug_check( ( (locs_i[0] >= n_rows) || (locs_i[1] >= n_cols) ), "SpMat::SpMat(): invalid row or column index" );
      
      arma_debug_check
        (
        ( (locs_i[1] < locs_im1[1]) || (locs_i[1] == locs_im1[1]  &&  locs_i[0] < locs_im1[0]) ),
        "SpMat::SpMat(): out of order points; either pass sort_locations = true, or sort points in column-major ordering"
        );
      
      if( (locs_i[1] == locs_im1[1]) && (locs_i[0] == locs_im1[0]) )
        {
        access::rw(values[count]) += vals[i];
        }
      else
        {
        count++;
        
        access::rw(values[count])      = vals[i];
        access::rw(row_indices[count]) = locs_i[0];
        
        access::rw(col_ptrs[ locs_i[1] + 1 ])++;
        }
      }
    }
  
  // Now fix the column pointers.
  for (uword i = 0; i < n_cols; ++i)
    {
    access::rw(col_ptrs[i + 1]) += col_ptrs[i];
    }
  }



template<typename eT>
inline
void
SpMat<eT>::mem_resize(const uword new_n_nonzero)
  {
  arma_extra_debug_sigprint();
  
  if(n_nonzero != new_n_nonzero)
    {
    if(new_n_nonzero == 0)
      {
      memory::release(values);
      memory::release(row_indices);
      
      access::rw(values)      = memory::acquire_chunked<eT>   (1);
      access::rw(row_indices) = memory::acquire_chunked<uword>(1);

      access::rw(     values[0]) = 0;
      access::rw(row_indices[0]) = 0;
      }
    else
      {
      // Figure out the actual amount of memory currently allocated
      // NOTE: this relies on memory::acquire_chunked() being used for the 'values' and 'row_indices' arrays
      const uword n_alloc = memory::enlarge_to_mult_of_chunksize(n_nonzero);
      
      if(n_alloc < new_n_nonzero)
        {
        eT*    new_values      = memory::acquire_chunked<eT>   (new_n_nonzero + 1);
        uword* new_row_indices = memory::acquire_chunked<uword>(new_n_nonzero + 1);
        
        if(n_nonzero > 0)
          {
          // Copy old elements.
          uword copy_len = std::min(n_nonzero, new_n_nonzero);
          
          arrayops::copy(new_values,      values,      copy_len);
          arrayops::copy(new_row_indices, row_indices, copy_len);
          }
        
        memory::release(values);
        memory::release(row_indices);
        
        access::rw(values)      = new_values;
        access::rw(row_indices) = new_row_indices;
        }
        
      // Set the "fake end" of the matrix by setting the last value and row
      // index to 0.  This helps the iterators work correctly.
      access::rw(     values[new_n_nonzero]) = 0;
      access::rw(row_indices[new_n_nonzero]) = 0;
      }
    
    access::rw(n_nonzero) = new_n_nonzero;
    }
  }



template<typename eT>
inline
void
SpMat<eT>::remove_zeros()
  {
  arma_extra_debug_sigprint();
  
  const uword old_n_nonzero = n_nonzero;
        uword new_n_nonzero = 0;
  
  const eT* old_values = values;
  
  for(uword i=0; i < old_n_nonzero; ++i)
    {
    new_n_nonzero += (old_values[i] != eT(0)) ? uword(1) : uword(0);
    }
  
  if(new_n_nonzero != old_n_nonzero)
    {
    if(new_n_nonzero == 0)  { init(n_rows, n_cols); return; }
    
    SpMat<eT> tmp(n_rows, n_cols);
    
    tmp.mem_resize(new_n_nonzero);
    
    uword new_index = 0;
    
    const_iterator it     = begin();
    const_iterator it_end = end();
    
    for(; it != it_end; ++it)
      {
      const eT val = eT(*it);
      
      if(val != eT(0))
        {
        access::rw(tmp.values[new_index])      = val;
        access::rw(tmp.row_indices[new_index]) = it.row();
        access::rw(tmp.col_ptrs[it.col() + 1])++;
        ++new_index;
        }
      }
    
    for(uword i=0; i < n_cols; ++i)
      {
      access::rw(tmp.col_ptrs[i + 1]) += tmp.col_ptrs[i];
      }
    
    steal_mem(tmp);
    }
  }



// Steal memory from another matrix.
template<typename eT>
inline
void
SpMat<eT>::steal_mem(SpMat<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  if(this != &x)
    {
    if(values     )  { memory::release(access::rw(values));      }
    if(row_indices)  { memory::release(access::rw(row_indices)); }
    if(col_ptrs   )  { memory::release(access::rw(col_ptrs));    }
    
    access::rw(n_rows)    = x.n_rows;
    access::rw(n_cols)    = x.n_cols;
    access::rw(n_elem)    = x.n_elem;
    access::rw(n_nonzero) = x.n_nonzero;
    
    access::rw(values)      = x.values;
    access::rw(row_indices) = x.row_indices;
    access::rw(col_ptrs)    = x.col_ptrs;
    
    // Set other matrix to empty.
    access::rw(x.n_rows)    = 0;
    access::rw(x.n_cols)    = 0;
    access::rw(x.n_elem)    = 0;
    access::rw(x.n_nonzero) = 0;
    
    access::rw(x.values)      = NULL;
    access::rw(x.row_indices) = NULL;
    access::rw(x.col_ptrs)    = NULL;
    }
  }



template<typename eT>
template<typename T1, typename Functor>
arma_hot
inline
void
SpMat<eT>::init_xform(const SpBase<eT,T1>& A, const Functor& func)
  {
  arma_extra_debug_sigprint();
  
  // if possible, avoid doing a copy and instead apply func to the generated elements
  if(SpProxy<T1>::Q_created_by_proxy == true)
    {
    (*this) = A.get_ref();
    
    const uword nnz = n_nonzero;
    
    eT* t_values = access::rwp(values);
    
    for(uword i=0; i < nnz; ++i)
      {
      t_values[i] = func(t_values[i]);
      }
    
    remove_zeros();
    }
  else
    {
    init_xform_mt(A.get_ref(), func);
    }
  }



template<typename eT>
template<typename eT2, typename T1, typename Functor>
arma_hot
inline
void
SpMat<eT>::init_xform_mt(const SpBase<eT2,T1>& A, const Functor& func)
  {
  arma_extra_debug_sigprint();
  
  const SpProxy<T1> P(A.get_ref());
  
  if( (P.is_alias(*this) == true) || (is_SpMat<typename SpProxy<T1>::stored_type>::value == true) )
    {
    // NOTE: unwrap_spmat will convert a submatrix to a matrix, which in effect takes care of aliasing with submatrices;
    // NOTE: however, when more delayed ops are implemented, more elaborate handling of aliasing will be necessary
    const unwrap_spmat<typename SpProxy<T1>::stored_type> tmp(P.Q);
    
    const SpMat<eT2>& x = tmp.M;
    
    if(void_ptr(this) != void_ptr(&x))
      {
      init(x.n_rows, x.n_cols);
      
      // values and row_indices may not be null.
      if(values != NULL)
        {
        memory::release(values);
        memory::release(row_indices);
        }
      
      access::rw(values)      = memory::acquire_chunked<eT>   (x.n_nonzero + 1);
      access::rw(row_indices) = memory::acquire_chunked<uword>(x.n_nonzero + 1);
      
      arrayops::copy(access::rwp(row_indices), x.row_indices, x.n_nonzero + 1);
      arrayops::copy(access::rwp(col_ptrs),    x.col_ptrs,    x.n_cols    + 1);
      
      access::rw(n_nonzero) = x.n_nonzero;
      }
    
    
    // initialise the elements array with a transformed version of the elements from x
    
    const uword nnz = n_nonzero;
    
    const eT2* x_values = x.values;
          eT*  t_values = access::rwp(values);
    
    for(uword i=0; i < nnz; ++i)
      {
      t_values[i] = func(x_values[i]);   // NOTE: func() must produce a value of type eT (ie. act as a convertor between eT2 and eT)
      }
    }
  else
    {
    init(P.get_n_rows(), P.get_n_cols());
    
    mem_resize(P.get_n_nonzero());
    
    typename SpProxy<T1>::const_iterator_type it     = P.begin();
    typename SpProxy<T1>::const_iterator_type it_end = P.end();
    
    while(it != it_end)
      {
      access::rw(row_indices[it.pos()]) = it.row();
      access::rw(values[it.pos()]) = func(*it);   // NOTE: func() must produce a value of type eT (ie. act as a convertor between eT2 and eT)
      ++access::rw(col_ptrs[it.col() + 1]);
      ++it;
      }
    
    // Now sum column pointers.
    for(uword c = 1; c <= n_cols; ++c)
      {
      access::rw(col_ptrs[c]) += col_ptrs[c - 1];
      }
    }
  
  remove_zeros();
  }



template<typename eT>
inline
typename SpMat<eT>::iterator
SpMat<eT>::begin()
  {
  return iterator(*this);
  }



template<typename eT>
inline
typename SpMat<eT>::const_iterator
SpMat<eT>::begin() const
  {
  return const_iterator(*this);
  }



template<typename eT>
inline
typename SpMat<eT>::iterator
SpMat<eT>::end()
  {
  return iterator(*this, 0, n_cols, n_nonzero);
  }



template<typename eT>
inline
typename SpMat<eT>::const_iterator
SpMat<eT>::end() const
  {
  return const_iterator(*this, 0, n_cols, n_nonzero);
  }



template<typename eT>
inline
typename SpMat<eT>::iterator
SpMat<eT>::begin_col(const uword col_num)
  {
  return iterator(*this, 0, col_num);
  }



template<typename eT>
inline
typename SpMat<eT>::const_iterator
SpMat<eT>::begin_col(const uword col_num) const
  {
  return const_iterator(*this, 0, col_num);
  }



template<typename eT>
inline
typename SpMat<eT>::iterator
SpMat<eT>::end_col(const uword col_num)
  {
  return iterator(*this, 0, col_num + 1);
  }



template<typename eT>
inline
typename SpMat<eT>::const_iterator
SpMat<eT>::end_col(const uword col_num) const
  {
  return const_iterator(*this, 0, col_num + 1);
  }



template<typename eT>
inline
typename SpMat<eT>::row_iterator
SpMat<eT>::begin_row(const uword row_num)
  {
  return row_iterator(*this, row_num, 0);
  }



template<typename eT>
inline
typename SpMat<eT>::const_row_iterator
SpMat<eT>::begin_row(const uword row_num) const
  {
  return const_row_iterator(*this, row_num, 0);
  }



template<typename eT>
inline
typename SpMat<eT>::row_iterator
SpMat<eT>::end_row()
  {
  return row_iterator(*this, n_nonzero);
  }



template<typename eT>
inline
typename SpMat<eT>::const_row_iterator
SpMat<eT>::end_row() const
  {
  return const_row_iterator(*this, n_nonzero);
  }



template<typename eT>
inline
typename SpMat<eT>::row_iterator
SpMat<eT>::end_row(const uword row_num)
  {
  return row_iterator(*this, row_num + 1, 0);
  }



template<typename eT>
inline
typename SpMat<eT>::const_row_iterator
SpMat<eT>::end_row(const uword row_num) const
  {
  return const_row_iterator(*this, row_num + 1, 0);
  }



template<typename eT>
inline
typename SpMat<eT>::row_col_iterator
SpMat<eT>::begin_row_col()
  {
  return begin();
  }



template<typename eT>
inline
typename SpMat<eT>::const_row_col_iterator
SpMat<eT>::begin_row_col() const
  {
  return begin();
  }



template<typename eT>
inline typename SpMat<eT>::row_col_iterator
SpMat<eT>::end_row_col()
  {
  return end();
  }



template<typename eT>
inline
typename SpMat<eT>::const_row_col_iterator
SpMat<eT>::end_row_col() const
  {
  return end();
  }



template<typename eT>
inline
void
SpMat<eT>::clear()
  {
  (*this).reset();
  }



template<typename eT>
inline
bool
SpMat<eT>::empty() const
  {
  return (n_elem == 0);
  }



template<typename eT>
inline
uword
SpMat<eT>::size() const
  {
  return n_elem;
  }



template<typename eT>
inline
arma_hot
arma_warn_unused
SpValProxy<SpMat<eT> >
SpMat<eT>::get_value(const uword i)
  {
  // First convert to the actual location.
  uword lcol = i / n_rows; // Integer division.
  uword lrow = i % n_rows;

  return get_value(lrow, lcol);
  }



template<typename eT>
inline
arma_hot
arma_warn_unused
eT
SpMat<eT>::get_value(const uword i) const
  {
  // First convert to the actual location.
  uword lcol = i / n_rows; // Integer division.
  uword lrow = i % n_rows;

  return get_value(lrow, lcol);
  }



template<typename eT>
inline
arma_hot
arma_warn_unused
SpValProxy<SpMat<eT> >
SpMat<eT>::get_value(const uword in_row, const uword in_col)
  {
  const uword colptr      = col_ptrs[in_col];
  const uword next_colptr = col_ptrs[in_col + 1];

  // Step through the row indices to see if our element exists.
  for (uword i = colptr; i < next_colptr; ++i)
    {
    const uword row_index = row_indices[i];
    
    // First check that we have not stepped past it.
    if (in_row < row_index) // If we have, then it doesn't exist: return 0.
      {
      return SpValProxy<SpMat<eT> >(in_row, in_col, *this); // Proxy for a zero value.
      }

    // Now check if we are at the correct place.
    if (in_row == row_index) // If we are, return a reference to the value.
      {
      return SpValProxy<SpMat<eT> >(in_row, in_col, *this, &access::rw(values[i]));
      }

    }

  // We did not find it, so it does not exist: return 0.
  return SpValProxy<SpMat<eT> >(in_row, in_col, *this);
  }



template<typename eT>
inline
arma_hot
arma_warn_unused
eT
SpMat<eT>::get_value(const uword in_row, const uword in_col) const
  {
  const uword colptr      = col_ptrs[in_col];
  const uword next_colptr = col_ptrs[in_col + 1];
  
  // Step through the row indices to see if our element exists.
  for (uword i = colptr; i < next_colptr; ++i)
    {
    const uword row_index = row_indices[i];
    
    // First check that we have not stepped past it.
    if (in_row < row_index) // If we have, then it doesn't exist: return 0.
      {
      return eT(0);
      }
    
    // Now check if we are at the correct place.
    if (in_row == row_index) // If we are, return the value.
      {
      return values[i];
      }
    }
  
  // We did not find it, so it does not exist: return 0.
  return eT(0);
  }



/**
 * Given the index representing which of the nonzero values this is, return its
 * actual location, either in row/col or just the index.
 */
template<typename eT>
arma_hot
arma_inline
arma_warn_unused
uword
SpMat<eT>::get_position(const uword i) const
  {
  uword lrow, lcol;
  
  get_position(i, lrow, lcol);
  
  // Assemble the row/col into the element's location in the matrix.
  return (lrow + n_rows * lcol);
  }



template<typename eT>
arma_hot
arma_inline
void
SpMat<eT>::get_position(const uword i, uword& row_of_i, uword& col_of_i) const
  {
  arma_debug_check((i >= n_nonzero), "SpMat::get_position(): index out of bounds");
  
  col_of_i = 0;
  while (col_ptrs[col_of_i + 1] <= i)
    {
    col_of_i++;
    }
  
  row_of_i = row_indices[i];
  
  return;
  }



/**
 * Add an element at the given position, and return a reference to it.  The
 * element will be set to 0 (unless otherwise specified).  If the element
 * already exists, its value will be overwritten.
 *
 * @param in_row Row of new element.
 * @param in_col Column of new element.
 * @param in_val Value to set new element to (default 0.0).
 */
template<typename eT>
inline
arma_hot
arma_warn_unused
eT&
SpMat<eT>::add_element(const uword in_row, const uword in_col, const eT val)
  {
  arma_extra_debug_sigprint();
  
  // We will assume the new element does not exist and begin the search for
  // where to insert it.  If we find that it already exists, we will then
  // overwrite it.
  uword colptr      = col_ptrs[in_col    ];
  uword next_colptr = col_ptrs[in_col + 1];
  
  uword pos = colptr; // The position in the matrix of this value.
  
  if (colptr != next_colptr)
    {
    // There are other elements in this column, so we must find where this
    // element will fit as compared to those.
    while (pos < next_colptr && in_row > row_indices[pos])
      {
      pos++;
      }
    
    // We aren't inserting into the last position, so it is still possible
    // that the element may exist.
    if (pos != next_colptr && row_indices[pos] == in_row)
      {
      // It already exists.  Then, just overwrite it.
      access::rw(values[pos]) = val;
      
      return access::rw(values[pos]);
      }
    }
  
  
  // 
  // Element doesn't exist, so we have to insert it
  // 
  
  // We have to update the rest of the column pointers.
  for (uword i = in_col + 1; i < n_cols + 1; i++)
    {
    access::rw(col_ptrs[i])++; // We are only inserting one new element.
    }
  
  
  // Figure out the actual amount of memory currently allocated
  // NOTE: this relies on memory::acquire_chunked() being used for the 'values' and 'row_indices' arrays
  const uword n_alloc = memory::enlarge_to_mult_of_chunksize(n_nonzero + 1);
  
  // If possible, avoid time-consuming memory allocation 
  if(n_alloc > (n_nonzero + 1))
    {
    arrayops::copy_backwards(access::rwp(values)      + pos + 1, values      + pos, (n_nonzero - pos) + 1);
    arrayops::copy_backwards(access::rwp(row_indices) + pos + 1, row_indices + pos, (n_nonzero - pos) + 1);
    
    // Insert the new element.
    access::rw(values[pos])      = val;
    access::rw(row_indices[pos]) = in_row;
    
    access::rw(n_nonzero)++;
    }
  else
    {
    const uword old_n_nonzero = n_nonzero;
    
    access::rw(n_nonzero)++; // Add to count of nonzero elements.
    
    // Allocate larger memory.
    eT*    new_values      = memory::acquire_chunked<eT>   (n_nonzero + 1);
    uword* new_row_indices = memory::acquire_chunked<uword>(n_nonzero + 1);
    
    // Copy things over, before the new element.
    if (pos > 0)
      {
      arrayops::copy(new_values,      values,      pos);
      arrayops::copy(new_row_indices, row_indices, pos);
      }
    
    // Insert the new element.
    new_values[pos]      = val;
    new_row_indices[pos] = in_row;
    
    // Copy the rest of things over (including the extra element at the end).
    arrayops::copy(new_values      + pos + 1, values      + pos, (old_n_nonzero - pos) + 1);
    arrayops::copy(new_row_indices + pos + 1, row_indices + pos, (old_n_nonzero - pos) + 1);
    
    // Assign new pointers.
    memory::release(values);
    memory::release(row_indices);

    access::rw(values)      = new_values;
    access::rw(row_indices) = new_row_indices;
    }
  
  return access::rw(values[pos]);
  }



/**
 * Delete an element at the given position.
 *
 * @param in_row Row of element to be deleted.
 * @param in_col Column of element to be deleted.
 */
template<typename eT>
inline
arma_hot
void
SpMat<eT>::delete_element(const uword in_row, const uword in_col)
  {
  arma_extra_debug_sigprint();
  
  // We assume the element exists (although... it may not) and look for its
  // exact position.  If it doesn't exist... well, we don't need to do anything.
  uword colptr      = col_ptrs[in_col];
  uword next_colptr = col_ptrs[in_col + 1];
  
  if (colptr != next_colptr)
    {
    // There's at least one element in this column.
    // Let's see if we are one of them.
    for (uword pos = colptr; pos < next_colptr; pos++)
      {
      if (in_row == row_indices[pos])
        {
        const uword old_n_nonzero = n_nonzero;
        
        --access::rw(n_nonzero); // Remove one from the count of nonzero elements.
        
        // Found it.  Now remove it.
        
        // Figure out the actual amount of memory currently allocated and the actual amount that will be required
        // NOTE: this relies on memory::acquire_chunked() being used for the 'values' and 'row_indices' arrays
        
        const uword n_alloc     = memory::enlarge_to_mult_of_chunksize(old_n_nonzero + 1);
        const uword n_alloc_mod = memory::enlarge_to_mult_of_chunksize(n_nonzero     + 1);
        
        // If possible, avoid time-consuming memory allocation
        if(n_alloc_mod == n_alloc)
          {
          if (pos < n_nonzero)  // remember, we decremented n_nonzero
            {
            arrayops::copy_forwards(access::rwp(values)      + pos, values      + pos + 1, (n_nonzero - pos) + 1);
            arrayops::copy_forwards(access::rwp(row_indices) + pos, row_indices + pos + 1, (n_nonzero - pos) + 1);
            }
          }
        else
          {
          // Make new arrays.
          eT*    new_values      = memory::acquire_chunked<eT>   (n_nonzero + 1);
          uword* new_row_indices = memory::acquire_chunked<uword>(n_nonzero + 1);
          
          if (pos > 0)
            {
            arrayops::copy(new_values,      values,      pos);
            arrayops::copy(new_row_indices, row_indices, pos);
            }
          
          arrayops::copy(new_values      + pos, values      + pos + 1, (n_nonzero - pos) + 1);
          arrayops::copy(new_row_indices + pos, row_indices + pos + 1, (n_nonzero - pos) + 1);
          
          memory::release(values);
          memory::release(row_indices);
          
          access::rw(values)      = new_values;
          access::rw(row_indices) = new_row_indices;
          }
        
        // And lastly, update all the column pointers (decrement by one).
        for (uword i = in_col + 1; i < n_cols + 1; i++)
          {
          --access::rw(col_ptrs[i]); // We only removed one element.
          }
        
        return; // There is nothing left to do.
        }
      }
    }
  
  return; // The element does not exist, so there's nothing for us to do.
  }



#ifdef ARMA_EXTRA_SPMAT_MEAT
  #include ARMA_INCFILE_WRAP(ARMA_EXTRA_SPMAT_MEAT)
#endif



//! @}
