// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// Copyright (C) 2008-2014 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup diagmat_proxy
//! @{



template<typename T1>
class diagmat_proxy_default
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  inline
  diagmat_proxy_default(const T1& X)
    : P       ( X )
    , P_is_vec( (resolves_to_vector<T1>::value) || (P.get_n_rows() == 1) || (P.get_n_cols() == 1) )
    , P_is_col( T1::is_col || (P.get_n_cols() == 1) )
    , n_elem  ( P_is_vec ? P.get_n_elem() : (std::min)(P.get_n_elem(), P.get_n_rows()) )
    {
    arma_extra_debug_sigprint();
    
    arma_debug_check
      (
      (P_is_vec == false) && (P.get_n_rows() != P.get_n_cols()),
      "diagmat(): only vectors and square matrices are accepted"
      );
    }
  
  
  arma_inline
  elem_type
  operator[](const uword i) const
    {
    if(Proxy<T1>::prefer_at_accessor == false)
      {
      return P_is_vec ? P[i] : P.at(i,i);
      }
    else
      {
      if(P_is_vec)
        {
        return (P_is_col) ? P.at(i,0) : P.at(0,i);
        }
      else
        {
        return P.at(i,i);
        }
      }
    }
  
  
  arma_inline
  elem_type
  at(const uword row, const uword col) const
    {
    if(row == col)
      {
      if(Proxy<T1>::prefer_at_accessor == false)
        {
        return (P_is_vec) ? P[row] : P.at(row,row);
        }
      else
        {
        if(P_is_vec)
          {
          return (P_is_col) ? P.at(row,0) : P.at(0,row);
          }
        else
          {
          return P.at(row,row);
          }
        }
      }
    else
      {
      return elem_type(0);
      }
    }
  
  
  arma_inline bool is_alias(const Mat<elem_type>&) const { return false; }
  
  const Proxy<T1> P;
  const bool      P_is_vec;
  const bool      P_is_col;
  const uword     n_elem;
  };



template<typename T1>
class diagmat_proxy_fixed
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  inline
  diagmat_proxy_fixed(const T1& X)
    : P(X)
    {
    arma_extra_debug_sigprint();
    
    arma_debug_check
      (
      (P_is_vec == false) && (T1::n_rows != T1::n_cols),
      "diagmat(): only vectors and square matrices are accepted"
      );
    }
  
  
  arma_inline
  elem_type
  operator[](const uword i) const
    {
    return (P_is_vec) ? P[i] : P.at(i,i);
    }
  
  
  arma_inline
  elem_type
  at(const uword row, const uword col) const
    {
    if(row == col)
      {
      return (P_is_vec) ? P[row] : P.at(row,row);
      }
    else
      {
      return elem_type(0);
      }
    }
  
  arma_inline bool is_alias(const Mat<elem_type>& X) const { return (void_ptr(&X) == void_ptr(&P)); }
  
  const T1& P;
  
  static const bool  P_is_vec = (T1::n_rows == 1) || (T1::n_cols == 1);
  static const uword n_elem   = P_is_vec ? T1::n_elem : ( (T1::n_elem < T1::n_rows) ? T1::n_elem : T1::n_rows );
  };



template<typename T1, bool condition>
struct diagmat_proxy_redirect {};

template<typename T1>
struct diagmat_proxy_redirect<T1, false> { typedef diagmat_proxy_default<T1> result; };

template<typename T1>
struct diagmat_proxy_redirect<T1, true>  { typedef diagmat_proxy_fixed<T1>   result; };


template<typename T1>
class diagmat_proxy : public diagmat_proxy_redirect<T1, is_Mat_fixed<T1>::value >::result
  {
  public:
  inline diagmat_proxy(const T1& X)
    : diagmat_proxy_redirect< T1, is_Mat_fixed<T1>::value >::result(X)
    {
    }
  };



template<typename eT>
class diagmat_proxy< Mat<eT> >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  inline
  diagmat_proxy(const Mat<eT>& X)
    : P       ( X )
    , P_is_vec( (X.n_rows == 1) || (X.n_cols == 1) )
    , n_elem  ( P_is_vec ? X.n_elem : (std::min)(X.n_elem, X.n_rows) )
    {
    arma_extra_debug_sigprint();
    
    arma_debug_check
      (
      (P_is_vec == false) && (P.n_rows != P.n_cols),
      "diagmat(): only vectors and square matrices are accepted"
      );
    }
  
  arma_inline elem_type operator[] (const uword i)                    const { return P_is_vec ? P[i] : P.at(i,i);                                         }
  arma_inline elem_type at         (const uword row, const uword col) const { return (row == col) ? ( P_is_vec ? P[row] : P.at(row,row) ) : elem_type(0); }
  
  arma_inline bool is_alias(const Mat<eT>& X) const { return (void_ptr(&X) == void_ptr(&P)); }
  
  const Mat<eT>& P;
  const bool     P_is_vec;
  const uword    n_elem;
  };



template<typename eT>
class diagmat_proxy< Row<eT> >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  
  inline
  diagmat_proxy(const Row<eT>& X)
    : P(X)
    , n_elem(X.n_elem)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline elem_type operator[] (const uword i)                    const { return P[i];                                 }
  arma_inline elem_type at         (const uword row, const uword col) const { return (row == col) ? P[row] : elem_type(0); }
  
  arma_inline bool is_alias(const Mat<eT>& X) const { return (void_ptr(&X) == void_ptr(&P)); }
  
  static const bool P_is_vec = true;
  
  const Row<eT>& P;
  const uword    n_elem;
  };



template<typename eT>
class diagmat_proxy< Col<eT> >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  
  inline
  diagmat_proxy(const Col<eT>& X)
    : P(X)
    , n_elem(X.n_elem)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline elem_type operator[] (const uword i)                    const { return P[i];                                 }
  arma_inline elem_type at         (const uword row, const uword col) const { return (row == col) ? P[row] : elem_type(0); }
  
  arma_inline bool is_alias(const Mat<eT>& X) const { return (void_ptr(&X) == void_ptr(&P)); }
  
  static const bool P_is_vec = true;
  
  const Col<eT>& P;
  const uword    n_elem;
  };



template<typename eT>
class diagmat_proxy< subview_row<eT> >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  
  inline
  diagmat_proxy(const subview_row<eT>& X)
    : P(X)
    , n_elem(X.n_elem)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline elem_type operator[] (const uword i)                    const { return P[i];                                 }
  arma_inline elem_type at         (const uword row, const uword col) const { return (row == col) ? P[row] : elem_type(0); }
  
  arma_inline bool is_alias(const Mat<eT>& X) const { return (void_ptr(&X) == void_ptr(&(P.m))); }
  
  static const bool P_is_vec = true;
  
  const subview_row<eT>& P;
  const uword            n_elem;
  };



template<typename eT>
class diagmat_proxy< subview_col<eT> >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  
  inline
  diagmat_proxy(const subview_col<eT>& X)
    : P(X)
    , n_elem(X.n_elem)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline elem_type operator[] (const uword i)                    const { return P[i];                                 }
  arma_inline elem_type at         (const uword row, const uword col) const { return (row == col) ? P[row] : elem_type(0); }
  
  arma_inline bool is_alias(const Mat<eT>& X) const { return (void_ptr(&X) == void_ptr(&(P.m))); }
  
  static const bool P_is_vec = true;
  
  const subview_col<eT>& P;
  const uword            n_elem;
  };



//
//
//



template<typename T1>
class diagmat_proxy_check_default
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  inline
  diagmat_proxy_check_default(const T1& X, const Mat<typename T1::elem_type>&)
    : P(X)
    , P_is_vec( (resolves_to_vector<T1>::value) || (P.n_rows == 1) || (P.n_cols == 1) )
    , n_elem( P_is_vec ? P.n_elem : (std::min)(P.n_elem, P.n_rows) )
    {
    arma_extra_debug_sigprint();
    
    arma_debug_check
      (
      (P_is_vec == false) && (P.n_rows != P.n_cols),
      "diagmat(): only vectors and square matrices are accepted"
      );
    }
  
  arma_inline elem_type operator[] (const uword i)                    const { return P_is_vec ? P[i] : P.at(i,i);                                         }
  arma_inline elem_type at         (const uword row, const uword col) const { return (row == col) ? ( P_is_vec ? P[row] : P.at(row,row) ) : elem_type(0); }
  
  const Mat<elem_type> P;
  const bool           P_is_vec;
  const uword          n_elem;
  };



template<typename T1>
class diagmat_proxy_check_fixed
  {
  public:
  
  typedef typename T1::elem_type                   eT;
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  inline
  diagmat_proxy_check_fixed(const T1& X, const Mat<eT>& out)
    : P( const_cast<eT*>(X.memptr()), T1::n_rows, T1::n_cols, (&X == &out), false )
    {
    arma_extra_debug_sigprint();
    
    arma_debug_check
      (
      (P_is_vec == false) && (T1::n_rows != T1::n_cols),
      "diagmat(): only vectors and square matrices are accepted"
      );
    }
  
  
  arma_inline eT operator[] (const uword i)                    const { return P_is_vec ? P[i] : P.at(i,i);                                         }
  arma_inline eT at         (const uword row, const uword col) const { return (row == col) ? ( P_is_vec ? P[row] : P.at(row,row) ) : elem_type(0); }
  
  const Mat<eT> P;  // TODO: why not just store X directly as T1& ?  test with fixed size vectors and matrices
  
  static const bool  P_is_vec = (T1::n_rows == 1) || (T1::n_cols == 1);
  static const uword n_elem   = P_is_vec ? T1::n_elem : ( (T1::n_elem < T1::n_rows) ? T1::n_elem : T1::n_rows );
  };



template<typename T1, bool condition>
struct diagmat_proxy_check_redirect {};

template<typename T1>
struct diagmat_proxy_check_redirect<T1, false> { typedef diagmat_proxy_check_default<T1> result; };

template<typename T1>
struct diagmat_proxy_check_redirect<T1, true>  { typedef diagmat_proxy_check_fixed<T1>   result; };


template<typename T1>
class diagmat_proxy_check : public diagmat_proxy_check_redirect<T1, is_Mat_fixed<T1>::value >::result
  {
  public:
  inline diagmat_proxy_check(const T1& X, const Mat<typename T1::elem_type>& out)
    : diagmat_proxy_check_redirect< T1, is_Mat_fixed<T1>::value >::result(X, out)
    {
    }
  };



template<typename eT>
class diagmat_proxy_check< Mat<eT> >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  
  inline
  diagmat_proxy_check(const Mat<eT>& X, const Mat<eT>& out)
    : P_local ( (&X == &out) ? new Mat<eT>(X) : 0  )
    , P       ( (&X == &out) ? (*P_local)     : X  )
    , P_is_vec( (P.n_rows == 1) || (P.n_cols == 1) )
    , n_elem  ( P_is_vec ? P.n_elem : (std::min)(P.n_elem, P.n_rows) )
    {
    arma_extra_debug_sigprint();
    
    arma_debug_check
      (
      (P_is_vec == false) && (P.n_rows != P.n_cols),
      "diagmat(): only vectors and square matrices are accepted"
      );
    }
  
  inline ~diagmat_proxy_check()
    {
    if(P_local) { delete P_local; }
    }
  
  arma_inline elem_type operator[] (const uword i)                    const { return P_is_vec ? P[i] : P.at(i,i);                                         }
  arma_inline elem_type at         (const uword row, const uword col) const { return (row == col) ? ( P_is_vec ? P[row] : P.at(row,row) ) : elem_type(0); }
  
  const Mat<eT>* P_local;
  const Mat<eT>& P;
  const bool     P_is_vec;
  const uword    n_elem;
  };



template<typename eT>
class diagmat_proxy_check< Row<eT> >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  inline
  diagmat_proxy_check(const Row<eT>& X, const Mat<eT>& out)
    : P_local ( (&X == reinterpret_cast<const Row<eT>*>(&out)) ? new Row<eT>(X) : 0 )
    , P       ( (&X == reinterpret_cast<const Row<eT>*>(&out)) ? (*P_local)     : X )
    , n_elem  (X.n_elem)
    {
    arma_extra_debug_sigprint();
    }
  
  inline ~diagmat_proxy_check()
    {
    if(P_local) { delete P_local; }
    }
  
  arma_inline elem_type operator[] (const uword i)                    const { return P[i];                                 }
  arma_inline elem_type at         (const uword row, const uword col) const { return (row == col) ? P[row] : elem_type(0); }
  
  static const bool P_is_vec = true;
  
  const Row<eT>* P_local;
  const Row<eT>& P;
  const uword    n_elem;
  };



template<typename eT>
class diagmat_proxy_check< Col<eT> >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  inline
  diagmat_proxy_check(const Col<eT>& X, const Mat<eT>& out)
    : P_local ( (&X == reinterpret_cast<const Col<eT>*>(&out)) ? new Col<eT>(X) : 0 )
    , P       ( (&X == reinterpret_cast<const Col<eT>*>(&out)) ? (*P_local)     : X )
    , n_elem  (X.n_elem)
    {
    arma_extra_debug_sigprint();
    }
  
  inline ~diagmat_proxy_check()
    {
    if(P_local) { delete P_local; }
    }
  
  arma_inline elem_type operator[] (const uword i)                    const { return P[i];                                 }
  arma_inline elem_type at         (const uword row, const uword col) const { return (row == col) ? P[row] : elem_type(0); }
  
  static const bool P_is_vec = true;
  
  const Col<eT>* P_local;
  const Col<eT>& P;
  const uword    n_elem;
  };



template<typename eT>
class diagmat_proxy_check< subview_row<eT> >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  inline
  diagmat_proxy_check(const subview_row<eT>& X, const Mat<eT>&)
    : P       ( X )
    , n_elem  ( X.n_elem )
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline elem_type operator[] (const uword i)                    const { return P[i];                                 }
  arma_inline elem_type at         (const uword row, const uword col) const { return (row == col) ? P[row] : elem_type(0); }
  
  static const bool P_is_vec = true;
  
  const Row<eT> P;
  const uword   n_elem;
  };



template<typename eT>
class diagmat_proxy_check< subview_col<eT> >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  inline
  diagmat_proxy_check(const subview_col<eT>& X, const Mat<eT>& out)
    : P     ( const_cast<eT*>(X.colptr(0)), X.n_rows, (&(X.m) == &out), false )
    , n_elem( X.n_elem )
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline elem_type operator[] (const uword i)                    const { return P[i];                                 }
  arma_inline elem_type at         (const uword row, const uword col) const { return (row == col) ? P[row] : elem_type(0); }
  
  static const bool P_is_vec = true;
  
  const Col<eT> P;
  const uword   n_elem;
  };



//! @}
