// Copyright (C) 2013-2014 Conrad Sanderson
// Copyright (C) 2013-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup SizeMat
//! @{



inline
SizeMat::SizeMat(const uword in_n_rows, const uword in_n_cols)
  : n_rows(in_n_rows)
  , n_cols(in_n_cols)
  {
  arma_extra_debug_sigprint();
  }



// inline
// SizeMat::operator SizeCube () const 
//   {
//   return SizeCube(n_rows, n_cols, 1);
//   }



inline
bool
SizeMat::operator==(const SizeMat& s) const
  {
  if(n_rows != s.n_rows)  { return false; }
  
  if(n_cols != s.n_cols)  { return false; }
  
  return true;
  }



inline
bool
SizeMat::operator!=(const SizeMat& s) const
  {
  if(n_rows != s.n_rows)  { return true; }
  
  if(n_cols != s.n_cols)  { return true; }
  
  return false;
  }



inline
bool
SizeMat::operator==(const SizeCube& s) const
  {
  if(n_rows   != s.n_rows  )  { return false; }
  
  if(n_cols   != s.n_cols  )  { return false; }
  
  if(uword(1) != s.n_slices)  { return false; }
  
  return true;
  }



inline
bool
SizeMat::operator!=(const SizeCube& s) const
  {
  if(n_rows   != s.n_rows  )  { return true; }
  
  if(n_cols   != s.n_cols  )  { return true; }
  
  if(uword(1) != s.n_slices)  { return true; }
  
  return false;
  }



inline
void
SizeMat::print(const std::string extra_text) const
  {
  arma_extra_debug_sigprint();
  
  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = ARMA_DEFAULT_OSTREAM.width();
    
    ARMA_DEFAULT_OSTREAM << extra_text << ' ';
  
    ARMA_DEFAULT_OSTREAM.width(orig_width);
    }
  
  arma_ostream::print(ARMA_DEFAULT_OSTREAM, *this);
  }



inline
void
SizeMat::print(std::ostream& user_stream, const std::string extra_text) const
  {
  arma_extra_debug_sigprint();
  
  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = user_stream.width();
    
    user_stream << extra_text << ' ';
    
    user_stream.width(orig_width);
    }
  
  arma_ostream::print(user_stream, *this);
  }



//! @}
