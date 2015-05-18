// Copyright (C) 2013-2014 Conrad Sanderson
// Copyright (C) 2013-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup SizeMat
//! @{



class SizeMat
  {
  public:
  
  const uword n_rows;
  const uword n_cols;
  
  inline SizeMat(const uword in_n_rows = 0, const uword in_n_cols = 0);
  
  // inline operator SizeCube () const;
  
  inline bool operator==(const SizeMat& s) const;
  inline bool operator!=(const SizeMat& s) const;
  
  inline bool operator==(const SizeCube& s) const;
  inline bool operator!=(const SizeCube& s) const;
  
  inline void print(const std::string extra_text = "") const;
  inline void print(std::ostream& user_stream, const std::string extra_text = "") const;
  };



//! @}
