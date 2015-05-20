// Copyright (C) 2008-2015 Conrad Sanderson
// Copyright (C) 2008-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup subview_cube
//! @{


//! Class for storing data required to construct or apply operations to a subcube
//! (i.e. where the subcube starts and ends as well as a reference/pointer to the original cube),
template<typename eT>
class subview_cube : public BaseCube<eT, subview_cube<eT> >
  {
  public:    
  
  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  arma_aligned const Cube<eT>& m;
  
  const uword aux_row1;
  const uword aux_col1;
  const uword aux_slice1;
  
  const uword n_rows;
  const uword n_cols;
  const uword n_elem_slice;
  const uword n_slices;
  const uword n_elem;
  
  
  protected:
  
  arma_inline subview_cube(const Cube<eT>& in_m, const uword in_row1, const uword in_col1, const uword in_slice1, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  
  
  public:
  
  inline ~subview_cube();
  
  inline void operator=  (const eT val);
  inline void operator+= (const eT val);
  inline void operator-= (const eT val);
  inline void operator*= (const eT val);
  inline void operator/= (const eT val);
  
  // deliberately returning void
  template<typename T1> inline void operator=  (const BaseCube<eT,T1>& x);
  template<typename T1> inline void operator+= (const BaseCube<eT,T1>& x);
  template<typename T1> inline void operator-= (const BaseCube<eT,T1>& x);
  template<typename T1> inline void operator%= (const BaseCube<eT,T1>& x);
  template<typename T1> inline void operator/= (const BaseCube<eT,T1>& x);
  
  inline void operator=  (const subview_cube& x);
  inline void operator+= (const subview_cube& x);
  inline void operator-= (const subview_cube& x);
  inline void operator%= (const subview_cube& x);
  inline void operator/= (const subview_cube& x);
  
  template<typename T1> inline void operator=  (const Base<eT,T1>& x);
  template<typename T1> inline void operator+= (const Base<eT,T1>& x);
  template<typename T1> inline void operator-= (const Base<eT,T1>& x);
  template<typename T1> inline void operator%= (const Base<eT,T1>& x);
  template<typename T1> inline void operator/= (const Base<eT,T1>& x);

  inline static void       extract(Cube<eT>& out, const subview_cube& in);
  inline static void  plus_inplace(Cube<eT>& out, const subview_cube& in);
  inline static void minus_inplace(Cube<eT>& out, const subview_cube& in);
  inline static void schur_inplace(Cube<eT>& out, const subview_cube& in);
  inline static void   div_inplace(Cube<eT>& out, const subview_cube& in);
  
  inline static void       extract(Mat<eT>& out, const subview_cube& in);
  inline static void  plus_inplace(Mat<eT>& out, const subview_cube& in);
  inline static void minus_inplace(Mat<eT>& out, const subview_cube& in);
  inline static void schur_inplace(Mat<eT>& out, const subview_cube& in);
  inline static void   div_inplace(Mat<eT>& out, const subview_cube& in);
  
  template<typename functor> inline void transform(functor F);
  template<typename functor> inline void     imbue(functor F);
  
  inline void fill(const eT val);
  inline void zeros();
  inline void ones();
  inline void randu();
  inline void randn();
  
  inline arma_warn_unused bool is_finite() const;

  inline arma_warn_unused eT min() const;
  inline arma_warn_unused eT max() const;
  
  inline eT  at_alt    (const uword i) const;
  
  inline eT& operator[](const uword i);
  inline eT  operator[](const uword i) const;
  
  inline eT& operator()(const uword i);
  inline eT  operator()(const uword i) const;
  
  arma_inline eT& operator()(const uword in_row, const uword in_col, const uword in_slice);
  arma_inline eT  operator()(const uword in_row, const uword in_col, const uword in_slice) const;
  
  arma_inline eT&         at(const uword in_row, const uword in_col, const uword in_slice);
  arma_inline eT          at(const uword in_row, const uword in_col, const uword in_slice) const;
  
  arma_inline       eT* slice_colptr(const uword in_slice, const uword in_col);
  arma_inline const eT* slice_colptr(const uword in_slice, const uword in_col) const;
  
  inline bool check_overlap(const subview_cube& x) const;
  inline bool check_overlap(const Mat<eT>&      x) const;
  
  
  private:
  
  friend class  Mat<eT>;
  friend class Cube<eT>;
  
  subview_cube();
  };



//! @}
