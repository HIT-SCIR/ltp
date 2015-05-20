// Copyright (C) 2011-2012 Ryan Curtin
// Copyright (C) 2011 Matthew Amidon
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup SpRow
//! @{

//! Class for sparse row vectors (sparse matrices with only one row)

template<typename eT>
class SpRow : public SpMat<eT>
  {
  public:

  typedef eT                                elem_type;
  typedef typename get_pod_type<eT>::result pod_type;
  
  static const bool is_row = true;
  static const bool is_col = false;
  
  
  inline          SpRow();
  inline explicit SpRow(const uword N);
  inline          SpRow(const uword in_rows, const uword in_cols);
  
  inline                  SpRow(const char*        text);
  inline const SpRow& operator=(const char*        text);
  
  inline                  SpRow(const std::string& text);
  inline const SpRow& operator=(const std::string& text);
  
  inline const SpRow& operator=(const eT val);
  
  template<typename T1> inline                  SpRow(const Base<eT,T1>& X);
  template<typename T1> inline const SpRow& operator=(const Base<eT,T1>& X);
  
  template<typename T1> inline                  SpRow(const SpBase<eT,T1>& X);
  template<typename T1> inline const SpRow& operator=(const SpBase<eT,T1>& X);
  
  template<typename T1, typename T2>
  inline explicit SpRow(const SpBase<pod_type,T1>& A, const SpBase<pod_type,T2>& B);
  
  inline void shed_col (const uword col_num);
  inline void shed_cols(const uword in_col1, const uword in_col2);
  
  // inline void insert_cols(const uword col_num, const uword N, const bool set_to_zero = true);
  
  
  typedef typename SpMat<eT>::iterator       row_iterator;
  typedef typename SpMat<eT>::const_iterator const_row_iterator;
  
  inline       row_iterator begin_row(const uword row_num = 0);
  inline const_row_iterator begin_row(const uword row_num = 0) const;
  
  inline       row_iterator end_row(const uword row_num = 0);
  inline const_row_iterator end_row(const uword row_num = 0) const;
  
  #ifdef ARMA_EXTRA_SPROW_PROTO
    #include ARMA_INCFILE_WRAP(ARMA_EXTRA_SPROW_PROTO)
  #endif
  };



//! @}
