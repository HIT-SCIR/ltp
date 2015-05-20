// Copyright (C) 2015 Conrad Sanderson
// Copyright (C) 2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup spdiagview
//! @{


//! Class for storing data required to extract and set the diagonals of a sparse matrix
template<typename eT>
class spdiagview : public Base<eT, spdiagview<eT> >
  {
  public:
  
  typedef eT                                elem_type;
  typedef typename get_pod_type<eT>::result pod_type;
  
  arma_aligned const SpMat<eT>& m;
  
  static const bool is_row = false;
  static const bool is_col = true;
  
  const uword row_offset;
  const uword col_offset;
  
  const uword n_rows;     // equal to n_elem
  const uword n_elem;
  
  static const uword n_cols = 1;
  
  
  protected:
  
  arma_inline spdiagview(const SpMat<eT>& in_m, const uword in_row_offset, const uword in_col_offset, const uword len);
  
  
  public:
  
  inline ~spdiagview();
  
  inline void operator=(const spdiagview& x);
  
  inline void operator+=(const eT val);
  inline void operator-=(const eT val);
  inline void operator*=(const eT val);
  inline void operator/=(const eT val);
  
  template<typename T1> inline void operator= (const Base<eT,T1>& x);
  template<typename T1> inline void operator+=(const Base<eT,T1>& x);
  template<typename T1> inline void operator-=(const Base<eT,T1>& x);
  template<typename T1> inline void operator%=(const Base<eT,T1>& x);
  template<typename T1> inline void operator/=(const Base<eT,T1>& x);
  
  template<typename T1> inline void operator= (const SpBase<eT,T1>& x);
  template<typename T1> inline void operator+=(const SpBase<eT,T1>& x);
  template<typename T1> inline void operator-=(const SpBase<eT,T1>& x);
  template<typename T1> inline void operator%=(const SpBase<eT,T1>& x);
  template<typename T1> inline void operator/=(const SpBase<eT,T1>& x);
  
  inline eT                      at_alt    (const uword ii) const;
  
  inline SpValProxy< SpMat<eT> > operator[](const uword ii);
  inline eT                      operator[](const uword ii) const;
  
  inline SpValProxy< SpMat<eT> >         at(const uword ii);
  inline eT                              at(const uword ii) const;
  
  inline SpValProxy< SpMat<eT> > operator()(const uword ii);
  inline eT                      operator()(const uword ii) const;
  
  inline SpValProxy< SpMat<eT> >         at(const uword in_n_row, const uword);
  inline eT                              at(const uword in_n_row, const uword) const;
  
  inline SpValProxy< SpMat<eT> > operator()(const uword in_n_row, const uword in_n_col);
  inline eT                      operator()(const uword in_n_row, const uword in_n_col) const;
  
  
  inline void fill(const eT val);
  inline void zeros();
  inline void ones();
  inline void randu();
  inline void randn();
    
  inline static void extract(Mat<eT>& out, const spdiagview& in);
  
  
  private:
  
  friend class SpMat<eT>;
  spdiagview();
  };


//! @}
