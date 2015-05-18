// Copyright (C) 2011-2012 Ryan Curtin
// Copyright (C) 2011 Matthew Amidon
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup SpCol
//! @{

//! Class for sparse column vectors (matrices with only one column)

template<typename eT>
class SpCol : public SpMat<eT>
  {
  public:
  
  typedef eT                                elem_type;
  typedef typename get_pod_type<eT>::result pod_type;
  
  static const bool is_row = false;
  static const bool is_col = true;
  
  
  inline          SpCol();
  inline explicit SpCol(const uword n_elem);
  inline          SpCol(const uword in_rows, const uword in_cols);
  
  inline                  SpCol(const char*        text);
  inline const SpCol& operator=(const char*        text);
  
  inline                  SpCol(const std::string& text);
  inline const SpCol& operator=(const std::string& text);
  
  inline const SpCol& operator=(const eT val);
  
  template<typename T1> inline                  SpCol(const Base<eT,T1>& X);
  template<typename T1> inline const SpCol& operator=(const Base<eT,T1>& X);
  
  template<typename T1> inline                  SpCol(const SpBase<eT,T1>& X);
  template<typename T1> inline const SpCol& operator=(const SpBase<eT,T1>& X);
  
  template<typename T1, typename T2>
  inline explicit SpCol(const SpBase<pod_type,T1>& A, const SpBase<pod_type,T2>& B);
  
  inline void shed_row (const uword row_num);
  inline void shed_rows(const uword in_row1, const uword in_row2);
  
  // inline void insert_rows(const uword row_num, const uword N, const bool set_to_zero = true);
  
  
  typedef typename SpMat<eT>::iterator       row_iterator;
  typedef typename SpMat<eT>::const_iterator const_row_iterator;
  
  inline       row_iterator begin_row(const uword row_num = 0);
  inline const_row_iterator begin_row(const uword row_num = 0) const;
  
  inline       row_iterator end_row  (const uword row_num = 0);
  inline const_row_iterator end_row  (const uword row_num = 0) const;
  
  
  #ifdef ARMA_EXTRA_SPCOL_PROTO
    #include ARMA_INCFILE_WRAP(ARMA_EXTRA_SPCOL_PROTO)
  #endif
  };
