// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup subview_field
//! @{


//! Class for storing data required to construct or apply operations to a subfield
//! (i.e. where the subfield starts and ends as well as a reference/pointer to the original field),
template<typename oT>
class subview_field
  {
  public:  
  
  typedef oT object_type;
  
  const field<oT>& f;
  
  const uword aux_row1;
  const uword aux_col1;
  const uword aux_slice1;
  
  const uword n_rows;
  const uword n_cols;
  const uword n_slices;
  const uword n_elem;
  
  
  protected:
  
  arma_inline subview_field(const field<oT>& in_f, const uword in_row1, const uword in_col1, const uword in_n_rows, const uword in_n_cols);
  arma_inline subview_field(const field<oT>& in_f, const uword in_row1, const uword in_col1, const uword in_slice1, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  
  
  public:
  
  inline ~subview_field();
  
  inline void operator= (const field<oT>& x);
  inline void operator= (const subview_field& x);
  
  arma_inline       oT& operator[](const uword i);
  arma_inline const oT& operator[](const uword i) const;
  
  arma_inline       oT& operator()(const uword i);
  arma_inline const oT& operator()(const uword i) const;
  
  arma_inline       oT&         at(const uword row, const uword col);
  arma_inline const oT&         at(const uword row, const uword col) const;

  arma_inline       oT&         at(const uword row, const uword col, const uword slice);
  arma_inline const oT&         at(const uword row, const uword col, const uword slice) const;
  
  arma_inline       oT& operator()(const uword row, const uword col);
  arma_inline const oT& operator()(const uword row, const uword col) const;
  
  arma_inline       oT& operator()(const uword row, const uword col, const uword slice);
  arma_inline const oT& operator()(const uword row, const uword col, const uword slice) const;
  
  inline bool check_overlap(const subview_field& x) const;
  
  inline void print(const std::string extra_text = "") const;
  inline void print(std::ostream& user_stream, const std::string extra_text = "") const;
  
  
  inline static void extract(field<oT>& out, const subview_field& in);
  
  
  private:
  
  friend class field<oT>;
  
  
  subview_field();
  //subview_field(const subview_field&);
  };


//! @}
