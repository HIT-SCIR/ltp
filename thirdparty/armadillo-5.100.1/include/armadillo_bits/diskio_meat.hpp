// Copyright (C) 2008-2015 NICTA (www.nicta.com.au)
// Copyright (C) 2008-2015 Conrad Sanderson
// Copyright (C) 2009-2010 Ian Cullinan
// Copyright (C) 2012 Ryan Curtin
// Copyright (C) 2013 Szabolcs Horvat
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup diskio
//! @{


//! Generate the first line of the header used for saving matrices in text format.
//! Format: "ARMA_MAT_TXT_ABXYZ".
//! A is one of: I (for integral types) or F (for floating point types).
//! B is one of: U (for unsigned types), S (for signed types), N (for not applicable) or C (for complex types).
//! XYZ specifies the width of each element in terms of bytes, e.g. "008" indicates eight bytes.
template<typename eT>
inline
std::string
diskio::gen_txt_header(const Mat<eT>& x)
  {
  arma_type_check(( is_supported_elem_type<eT>::value == false ));

  arma_ignore(x);
  
  if(is_u8<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IU001");
    }
  else
  if(is_s8<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IS001");
    }
  else
  if(is_u16<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IU002");
    }
  else
  if(is_s16<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IS002");
    }
  else
  if(is_u32<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IU004");
    }
  else
  if(is_s32<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IS004");
    }
#if defined(ARMA_USE_U64S64)
  else
  if(is_u64<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IU008");
    }
  else
  if(is_s64<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IS008");
    }
#endif
#if defined(ARMA_ALLOW_LONG)
  else
  if(is_ulng_t_32<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IU004");
    }
  else
  if(is_slng_t_32<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IS004");
    }
  else
  if(is_ulng_t_64<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IU008");
    }
  else
  if(is_slng_t_64<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_IS008");
    }
#endif
  else
  if(is_float<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_FN004");
    }
  else
  if(is_double<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_FN008");
    }
  else
  if(is_complex_float<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_FC008");
    }
  else
  if(is_complex_double<eT>::value == true)
    {
    return std::string("ARMA_MAT_TXT_FC016");
    }
  else
    {
    return std::string();
    }
  
  }



//! Generate the first line of the header used for saving matrices in binary format.
//! Format: "ARMA_MAT_BIN_ABXYZ".
//! A is one of: I (for integral types) or F (for floating point types).
//! B is one of: U (for unsigned types), S (for signed types), N (for not applicable) or C (for complex types).
//! XYZ specifies the width of each element in terms of bytes, e.g. "008" indicates eight bytes.
template<typename eT>
inline
std::string
diskio::gen_bin_header(const Mat<eT>& x)
  {
  arma_type_check(( is_supported_elem_type<eT>::value == false ));
  
  arma_ignore(x);
  
  if(is_u8<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IU001");
    }
  else
  if(is_s8<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IS001");
    }
  else
  if(is_u16<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IU002");
    }
  else
  if(is_s16<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IS002");
    }
  else
  if(is_u32<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IU004");
    }
  else
  if(is_s32<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IS004");
    }
#if defined(ARMA_USE_U64S64)
  else
  if(is_u64<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IU008");
    }
  else
  if(is_s64<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IS008");
    }
#endif
#if defined(ARMA_ALLOW_LONG)
  else
  if(is_ulng_t_32<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IU004");
    }
  else
  if(is_slng_t_32<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IS004");
    }
  else
  if(is_ulng_t_64<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IU008");
    }
  else
  if(is_slng_t_64<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_IS008");
    }
#endif
  else
  if(is_float<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_FN004");
    }
  else
  if(is_double<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_FN008");
    }
  else
  if(is_complex_float<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_FC008");
    }
  else
  if(is_complex_double<eT>::value == true)
    {
    return std::string("ARMA_MAT_BIN_FC016");
    }
  else
    {
    return std::string();
    }
  
  }



//! Generate the first line of the header used for saving matrices in binary format.
//! Format: "ARMA_SPM_BIN_ABXYZ".
//! A is one of: I (for integral types) or F (for floating point types).
//! B is one of: U (for unsigned types), S (for signed types), N (for not applicable) or C (for complex types).
//! XYZ specifies the width of each element in terms of bytes, e.g. "008" indicates eight bytes.
template<typename eT>
inline
std::string
diskio::gen_bin_header(const SpMat<eT>& x)
  {
  arma_type_check(( is_supported_elem_type<eT>::value == false ));

  arma_ignore(x);

  if(is_u8<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IU001");
    }
  else
  if(is_s8<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IS001");
    }
  else
  if(is_u16<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IU002");
    }
  else
  if(is_s16<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IS002");
    }
  else
  if(is_u32<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IU004");
    }
  else
  if(is_s32<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IS004");
    }
#if defined(ARMA_USE_U64S64)
  else
  if(is_u64<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IU008");
    }
  else
  if(is_s64<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IS008");
    }
#endif
#if defined(ARMA_ALLOW_LONG)
  else
  if(is_ulng_t_32<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IU004");
    }
  else
  if(is_slng_t_32<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IS004");
    }
  else
  if(is_ulng_t_64<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IU008");
    }
  else
  if(is_slng_t_64<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_IS008");
    }
#endif
  else
  if(is_float<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_FN004");
    }
  else
  if(is_double<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_FN008");
    }
  else
  if(is_complex_float<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_FC008");
    }
  else
  if(is_complex_double<eT>::value == true)
    {
    return std::string("ARMA_SPM_BIN_FC016");
    }
  else
    {
    return std::string();
    }

  }


//! Generate the first line of the header used for saving cubes in text format.
//! Format: "ARMA_CUB_TXT_ABXYZ".
//! A is one of: I (for integral types) or F (for floating point types).
//! B is one of: U (for unsigned types), S (for signed types), N (for not applicable) or C (for complex types).
//! XYZ specifies the width of each element in terms of bytes, e.g. "008" indicates eight bytes.
template<typename eT>
inline
std::string
diskio::gen_txt_header(const Cube<eT>& x)
  {
  arma_type_check(( is_supported_elem_type<eT>::value == false ));
  
  arma_ignore(x);

  if(is_u8<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IU001");
    }
  else
  if(is_s8<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IS001");
    }
  else
  if(is_u16<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IU002");
    }
  else
  if(is_s16<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IS002");
    }
  else
  if(is_u32<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IU004");
    }
  else
  if(is_s32<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IS004");
    }
#if defined(ARMA_USE_U64S64)
  else
  if(is_u64<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IU008");
    }
  else
  if(is_s64<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IS008");
    }
#endif
#if defined(ARMA_ALLOW_LONG)
  else
  if(is_ulng_t_32<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IU004");
    }
  else
  if(is_slng_t_32<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IS004");
    }
  else
  if(is_ulng_t_64<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IU008");
    }
  else
  if(is_slng_t_64<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_IS008");
    }
#endif
  else
  if(is_float<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_FN004");
    }
  else
  if(is_double<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_FN008");
    }
  else
  if(is_complex_float<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_FC008");
    }
  else
  if(is_complex_double<eT>::value == true)
    {
    return std::string("ARMA_CUB_TXT_FC016");
    }
  else
    {
    return std::string();
    }
  
  }



//! Generate the first line of the header used for saving cubes in binary format.
//! Format: "ARMA_CUB_BIN_ABXYZ".
//! A is one of: I (for integral types) or F (for floating point types).
//! B is one of: U (for unsigned types), S (for signed types), N (for not applicable) or C (for complex types).
//! XYZ specifies the width of each element in terms of bytes, e.g. "008" indicates eight bytes.
template<typename eT>
inline
std::string
diskio::gen_bin_header(const Cube<eT>& x)
  {
  arma_type_check(( is_supported_elem_type<eT>::value == false ));
  
  arma_ignore(x);
  
  if(is_u8<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IU001");
    }
  else
  if(is_s8<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IS001");
    }
  else
  if(is_u16<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IU002");
    }
  else
  if(is_s16<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IS002");
    }
  else
  if(is_u32<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IU004");
    }
  else
  if(is_s32<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IS004");
    }
#if defined(ARMA_USE_U64S64)
  else
  if(is_u64<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IU008");
    }
  else
  if(is_s64<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IS008");
    }
#endif
#if defined(ARMA_ALLOW_LONG)
  else
  if(is_ulng_t_32<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IU004");
    }
  else
  if(is_slng_t_32<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IS004");
    }
  else
  if(is_ulng_t_64<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IU008");
    }
  else
  if(is_slng_t_64<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_IS008");
    }
#endif
  else
  if(is_float<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_FN004");
    }
  else
  if(is_double<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_FN008");
    }
  else
  if(is_complex_float<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_FC008");
    }
  else
  if(is_complex_double<eT>::value == true)
    {
    return std::string("ARMA_CUB_BIN_FC016");
    }
  else
    {
    return std::string();
    }
  
  }



inline
file_type
diskio::guess_file_type(std::istream& f)
  {
  arma_extra_debug_sigprint();
  
  f.clear();
  const std::fstream::pos_type pos1 = f.tellg();
  
  f.clear();
  f.seekg(0, ios::end);
  
  f.clear();
  const std::fstream::pos_type pos2 = f.tellg();
  
  const uword N = ( (pos1 >= 0) && (pos2 >= 0) ) ? uword(pos2 - pos1) : 0;
  
  f.clear();
  f.seekg(pos1);
  
  podarray<unsigned char> data(N);
  
  unsigned char* ptr = data.memptr();
  
  f.clear();
  f.read( reinterpret_cast<char*>(ptr), std::streamsize(N) );
  
  const bool load_okay = f.good();
  
  f.clear();
  f.seekg(pos1);
  
  bool has_binary  = false;
  bool has_comma   = false;
  bool has_bracket = false;
  
  if(load_okay == true)
    {
    uword i = 0;
    uword j = (N >= 2) ? 1 : 0;
    
    for(; j<N; i+=2, j+=2)
      {
      const unsigned char val_i = ptr[i];
      const unsigned char val_j = ptr[j];
      
      // the range checking can be made more elaborate
      if( ((val_i <= 8) || (val_i >= 123)) || ((val_j <= 8) || (val_j >= 123)) )
        {
        has_binary = true;
        break;
        }
      
      if( (val_i == ',') || (val_j == ',') )
        {
        has_comma = true;
        }
      
      if( (val_i == '(') || (val_j == '(') || (val_i == ')') || (val_j == ')') )
        {
        has_bracket = true;
        }
      }
    }
  else
    {
    return file_type_unknown;
    }
  
  if(has_binary)
    {
    return raw_binary;
    }
  
  if(has_comma && (has_bracket == false))
    {
    return csv_ascii;
    }
  
  return raw_ascii;
  }



inline
char
diskio::conv_to_hex_char(const u8 x)
  {
  char out;

  switch(x)
    {
    case  0: out = '0'; break;
    case  1: out = '1'; break;
    case  2: out = '2'; break;
    case  3: out = '3'; break;
    case  4: out = '4'; break;
    case  5: out = '5'; break;
    case  6: out = '6'; break;
    case  7: out = '7'; break;
    case  8: out = '8'; break;
    case  9: out = '9'; break;
    case 10: out = 'a'; break;
    case 11: out = 'b'; break;
    case 12: out = 'c'; break;
    case 13: out = 'd'; break;
    case 14: out = 'e'; break;
    case 15: out = 'f'; break;
    default: out = '-'; break;
    }

  return out;  
  }



inline
void
diskio::conv_to_hex(char* out, const u8 x)
  {
  const u8 a = x / 16;
  const u8 b = x - 16*a;

  out[0] = conv_to_hex_char(a);
  out[1] = conv_to_hex_char(b);
  }



//! Append a quasi-random string to the given filename.
//! The rand() function is deliberately not used,
//! as rand() has an internal state that changes
//! from call to call. Such states should not be
//! modified in scientific applications, where the
//! results should be reproducable and not affected 
//! by saving data.
inline
std::string
diskio::gen_tmp_name(const std::string& x)
  {
  const std::string* ptr_x     = &x;
  const u8*          ptr_ptr_x = reinterpret_cast<const u8*>(&ptr_x);
  
  const char* extra      = ".tmp_";
  const uword extra_size = 5;
  
  const uword tmp_size   = 2*sizeof(u8*) + 2*2;
        char  tmp[tmp_size];
  
  uword char_count = 0;
  
  for(uword i=0; i<sizeof(u8*); ++i)
    {
    conv_to_hex(&tmp[char_count], ptr_ptr_x[i]);
    char_count += 2;
    }
  
  const uword x_size = static_cast<uword>(x.size());
  u8 sum = 0;
  
  for(uword i=0; i<x_size; ++i)
    {
    sum = (sum + u8(x[i])) & 0xff;
    }
  
  conv_to_hex(&tmp[char_count], sum);
  char_count += 2;
  
  conv_to_hex(&tmp[char_count], u8(x_size));
  
  
  std::string out;
  out.resize(x_size + extra_size + tmp_size);
  
  
  for(uword i=0; i<x_size; ++i)
    {
    out[i] = x[i];
    }
  
  for(uword i=0; i<extra_size; ++i)
    {
    out[x_size + i] = extra[i];
    }
  
  for(uword i=0; i<tmp_size; ++i)
    {
    out[x_size + extra_size + i] = tmp[i];
    }
  
  return out;
  }



//! Safely rename a file.
//! Before renaming, test if we can write to the final file.
//! This should prevent:
//! (i)  overwriting files that are write protected,
//! (ii) overwriting directories.
inline
bool
diskio::safe_rename(const std::string& old_name, const std::string& new_name)
  {
  std::fstream f(new_name.c_str(), std::fstream::out | std::fstream::app);
  f.put(' ');
  
  bool save_okay = f.good();
  f.close();
  
  if(save_okay == true)
    {
    std::remove(new_name.c_str());
    
    const int mv_result = std::rename(old_name.c_str(), new_name.c_str());
    
    save_okay = (mv_result == 0);
    }
  
  return save_okay;
  }



template<typename eT>
inline
bool
diskio::convert_naninf(eT& val, const std::string& token)
  {
  // see if the token represents a NaN or Inf
  
  if( (token.length() == 3) || (token.length() == 4) )
    {
    const bool neg = (token[0] == '-');
    const bool pos = (token[0] == '+');
    
    const size_t offset = ( (neg || pos) && (token.length() == 4) ) ? 1 : 0;
    
    const std::string token2 = token.substr(offset, 3);
    
    if( (token2 == "inf") || (token2 == "Inf") || (token2 == "INF") )
      {
      val = neg ? -(Datum<eT>::inf) : Datum<eT>::inf;
      
      return true;
      }
    else
    if( (token2 == "nan") || (token2 == "Nan") || (token2 == "NaN") || (token2 == "NAN") )
      {
      val = neg ? -(Datum<eT>::nan) : Datum<eT>::nan;
      
      return true;
      }
    }
    
  return false;
  }



template<typename T>
inline
bool
diskio::convert_naninf(std::complex<T>& val, const std::string& token)
  {
  if( token.length() >= 5 )
    {
    std::stringstream ss( token.substr(1, token.length()-2) );  // strip '(' at the start and ')' at the end
    
    std::string token_real;
    std::string token_imag;
    
    std::getline(ss, token_real, ',');
    std::getline(ss, token_imag);
    
    std::stringstream ss_real(token_real);
    std::stringstream ss_imag(token_imag);
    
    T val_real = T(0);
    T val_imag = T(0);
    
    ss_real >> val_real;
    ss_imag >> val_imag;
    
    bool success_real = true;
    bool success_imag = true;
    
    if(ss_real.fail() == true)
      {
      success_real = diskio::convert_naninf( val_real, token_real );
      }
    
    if(ss_imag.fail() == true)
      {
      success_imag = diskio::convert_naninf( val_imag, token_imag );
      }
    
    val = std::complex<T>(val_real, val_imag);
    
    return (success_real && success_imag);
    }
  
  return false;
  }



//! Save a matrix as raw text (no header, human readable).
//! Matrices can be loaded in Matlab and Octave, as long as they don't have complex elements.
template<typename eT>
inline
bool
diskio::save_raw_ascii(const Mat<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::fstream f(tmp_name.c_str(), std::fstream::out);
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = diskio::save_raw_ascii(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



//! Save a matrix as raw text (no header, human readable).
//! Matrices can be loaded in Matlab and Octave, as long as they don't have complex elements.
template<typename eT>
inline
bool
diskio::save_raw_ascii(const Mat<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  uword cell_width;
  
  // TODO: need sane values for complex numbers
  
  if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )
    {
    f.setf(ios::scientific);
    f.precision(12);
    cell_width = 20;
    }
  
  for(uword row=0; row < x.n_rows; ++row)
    {
    for(uword col=0; col < x.n_cols; ++col)
      {
      f.put(' ');
      
      if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )
        {
        f.width(cell_width);
        }
      
      arma_ostream::print_elem(f, x.at(row,col), false);
      }
      
    f.put('\n');
    }
  
  return f.good();
  }



//! Save a matrix as raw binary (no header)
template<typename eT>
inline
bool
diskio::save_raw_binary(const Mat<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::ofstream f(tmp_name.c_str(), std::fstream::binary);
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = diskio::save_raw_binary(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



template<typename eT>
inline
bool
diskio::save_raw_binary(const Mat<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  f.write( reinterpret_cast<const char*>(x.mem), std::streamsize(x.n_elem*sizeof(eT)) );
  
  return f.good();
  }



//! Save a matrix in text format (human readable),
//! with a header that indicates the matrix type as well as its dimensions
template<typename eT>
inline
bool
diskio::save_arma_ascii(const Mat<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::ofstream f(tmp_name.c_str());
  
  bool save_okay = f.is_open();

  if(save_okay == true)  
    {
    save_okay = diskio::save_arma_ascii(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



//! Save a matrix in text format (human readable),
//! with a header that indicates the matrix type as well as its dimensions
template<typename eT>
inline
bool
diskio::save_arma_ascii(const Mat<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  const ios::fmtflags orig_flags = f.flags();
  
  f << diskio::gen_txt_header(x) << '\n';
  f << x.n_rows << ' ' << x.n_cols << '\n';
  
  uword cell_width;
  
  // TODO: need sane values for complex numbers
  
  if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )
    {
    f.setf(ios::scientific);
    f.precision(12);
    cell_width = 20;
    }
    
  for(uword row=0; row < x.n_rows; ++row)
    {
    for(uword col=0; col < x.n_cols; ++col)
      {
      f.put(' ');
      
      if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )        
        {
        f.width(cell_width);
        }
      
      arma_ostream::print_elem(f, x.at(row,col), false);
      }
    
    f.put('\n');
    }
  
  const bool save_okay = f.good();
  
  f.flags(orig_flags);
  
  return save_okay;
  }



//! Save a matrix in CSV text format (human readable)
template<typename eT>
inline
bool
diskio::save_csv_ascii(const Mat<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::ofstream f(tmp_name.c_str());
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)  
    {
    save_okay = diskio::save_csv_ascii(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



//! Save a matrix in CSV text format (human readable)
template<typename eT>
inline
bool
diskio::save_csv_ascii(const Mat<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  const ios::fmtflags orig_flags = f.flags();
  
  // TODO: need sane values for complex numbers
  
  if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )
    {
    f.setf(ios::scientific);
    f.precision(12);
    }
  
  uword x_n_rows = x.n_rows;
  uword x_n_cols = x.n_cols;
  
  for(uword row=0; row < x_n_rows; ++row)
    {
    for(uword col=0; col < x_n_cols; ++col)
      {
      arma_ostream::print_elem(f, x.at(row,col), false);
      
      if( col < (x_n_cols-1) )
        {
        f.put(',');
        }
      }
    
    f.put('\n');
    }
  
  const bool save_okay = f.good();
  
  f.flags(orig_flags);
  
  return save_okay;
  }



//! Save a matrix in binary format,
//! with a header that stores the matrix type as well as its dimensions
template<typename eT>
inline
bool
diskio::save_arma_binary(const Mat<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::ofstream f(tmp_name.c_str(), std::fstream::binary);
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = diskio::save_arma_binary(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



//! Save a matrix in binary format,
//! with a header that stores the matrix type as well as its dimensions
template<typename eT>
inline
bool
diskio::save_arma_binary(const Mat<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();

  f << diskio::gen_bin_header(x) << '\n';
  f << x.n_rows << ' ' << x.n_cols << '\n';
  
  f.write( reinterpret_cast<const char*>(x.mem), std::streamsize(x.n_elem*sizeof(eT)) );
  
  return f.good();
  }



//! Save a matrix as a PGM greyscale image
template<typename eT>
inline
bool
diskio::save_pgm_binary(const Mat<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::fstream f(tmp_name.c_str(), std::fstream::out | std::fstream::binary);
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = diskio::save_pgm_binary(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



//
// TODO:
// add functionality to save the image in a normalised format,
// i.e. scaled so that every value falls in the [0,255] range.

//! Save a matrix as a PGM greyscale image
template<typename eT>
inline
bool
diskio::save_pgm_binary(const Mat<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  f << "P5" << '\n';
  f << x.n_cols << ' ' << x.n_rows << '\n';
  f << 255 << '\n';
  
  const uword n_elem = x.n_rows * x.n_cols;
  podarray<u8> tmp(n_elem);
  
  uword i = 0;
  
  for(uword row=0; row < x.n_rows; ++row)
    {
    for(uword col=0; col < x.n_cols; ++col)
      {
      tmp[i] = u8( x.at(row,col) );  // TODO: add round() ?
      ++i;
      }
    }
  
  f.write(reinterpret_cast<const char*>(tmp.mem), std::streamsize(n_elem) );
  
  return f.good();
  }



//! Save a matrix as a PGM greyscale image
template<typename T>
inline
bool
diskio::save_pgm_binary(const Mat< std::complex<T> >& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const uchar_mat tmp = conv_to<uchar_mat>::from(x);
  
  return diskio::save_pgm_binary(tmp, final_name);
  }



//! Save a matrix as a PGM greyscale image
template<typename T>
inline
bool
diskio::save_pgm_binary(const Mat< std::complex<T> >& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  const uchar_mat tmp = conv_to<uchar_mat>::from(x);
  
  return diskio::save_pgm_binary(tmp, f);
  }



//! Save a matrix as part of a HDF5 file
template<typename eT>
inline 
bool 
diskio::save_hdf5_binary(const Mat<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_HDF5)
    {
    #if !defined(ARMA_PRINT_HDF5_ERRORS)
      {
      // Disable annoying HDF5 error messages.
      arma_H5Eset_auto(H5E_DEFAULT, NULL, NULL);
      }
    #endif
    
    bool save_okay = false;
    
    const std::string tmp_name = diskio::gen_tmp_name(final_name);
    
    // Set up the file according to HDF5's preferences  
    hid_t file = arma_H5Fcreate(tmp_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    // We need to create a dataset, datatype, and dataspace
    hsize_t dims[2];
    dims[1] = x.n_rows;
    dims[0] = x.n_cols;
    
    hid_t dataspace = arma_H5Screate_simple(2, dims, NULL);   // treat the matrix as a 2d array dataspace
    hid_t datatype  = hdf5_misc::get_hdf5_type<eT>();
    
    // If this returned something invalid, well, it's time to crash.
    arma_check(datatype == -1, "Mat::save(): unknown datatype for HDF5");
    
    // MATLAB forces the users to specify a name at save time for HDF5; Octave
    // will use the default of 'dataset' unless otherwise specified, so we will
    // use that.
    hid_t dataset = arma_H5Dcreate(file, "dataset", datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // H5Dwrite does not make a distinction between row-major and column-major;
    // it just writes the memory.  MATLAB and Octave store HDF5 matrices as
    // column-major, though, so we can save ours like that too and not need to
    // transpose.
    herr_t status = arma_H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, x.mem);
    save_okay = (status >= 0);
    
    arma_H5Dclose(dataset);
    arma_H5Tclose(datatype);
    arma_H5Sclose(dataspace);
    arma_H5Fclose(file);
    
    if(save_okay == true) { save_okay = diskio::safe_rename(tmp_name, final_name); }
    
    return save_okay;
    }
  #else
    {
    arma_ignore(x);
    arma_ignore(final_name);
    
    arma_stop("Mat::save(): use of HDF5 needs to be enabled");
    
    return false;
    }
  #endif
  }



//! Load a matrix as raw text (no header, human readable).
//! Can read matrices saved as text in Matlab and Octave.
//! NOTE: this is much slower than reading a file with a header.
template<typename eT>
inline
bool
diskio::load_raw_ascii(Mat<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();

  std::fstream f;
  f.open(name.c_str(), std::fstream::in);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_raw_ascii(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



//! Load a matrix as raw text (no header, human readable).
//! Can read matrices saved as text in Matlab and Octave.
//! NOTE: this is much slower than reading a file with a header.
template<typename eT>
inline
bool
diskio::load_raw_ascii(Mat<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  bool load_okay = f.good();
  
  f.clear();
  const std::fstream::pos_type pos1 = f.tellg();
  
  //
  // work out the size
  
  uword f_n_rows = 0;
  uword f_n_cols = 0;
  
  bool f_n_cols_found = false;
  
  std::string line_string;
  std::string token;
  
  std::stringstream line_stream;
  
  while( (f.good() == true) && (load_okay == true) )
    {
    std::getline(f, line_string);
    
    if(line_string.size() == 0)
      {
      break;
      }
    
    line_stream.clear();
    line_stream.str(line_string);
    
    uword line_n_cols = 0;
    
    while (line_stream >> token)
      {
      ++line_n_cols;
      }
    
    if(f_n_cols_found == false)
      {
      f_n_cols = line_n_cols;
      f_n_cols_found = true;
      }
    else
      {
      if(line_n_cols != f_n_cols)
        {
        err_msg = "inconsistent number of columns in ";
        load_okay = false;
        }
      }
    
    ++f_n_rows;
    }
    
  if(load_okay == true)
    {
    f.clear();
    f.seekg(pos1);
    
    x.set_size(f_n_rows, f_n_cols);
    
    std::stringstream ss;
    
    for(uword row=0; (row < x.n_rows) && (load_okay == true); ++row)
      {
      for(uword col=0; (col < x.n_cols) && (load_okay == true); ++col)
        {
        f >> token;
        
        ss.clear();
        ss.str(token);
        
        eT val = eT(0);
        ss >> val;
        
        if(ss.fail() == false)
          {
          x.at(row,col) = val;
          }
        else
          {
          const bool success = diskio::convert_naninf( x.at(row,col), token );
          
          if(success == false)
            {
            load_okay = false;
            err_msg = "couldn't interpret data in ";
            }
          }
        }
      }
    }
  
  
  // an empty file indicates an empty matrix
  if( (f_n_cols_found == false) && (load_okay == true) )
    {
    x.reset();
    }
  
  
  return load_okay;
  }



//! Load a matrix in binary format (no header);
//! the matrix is assumed to have one column
template<typename eT>
inline
bool
diskio::load_raw_binary(Mat<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::ifstream f;
  f.open(name.c_str(), std::fstream::binary);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_raw_binary(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



template<typename eT>
inline
bool
diskio::load_raw_binary(Mat<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  arma_ignore(err_msg);
  
  f.clear();
  const std::streampos pos1 = f.tellg();
  
  f.clear();
  f.seekg(0, ios::end);

  f.clear();
  const std::streampos pos2 = f.tellg();
  
  const uword N = ( (pos1 >= 0) && (pos2 >= 0) ) ? uword(pos2 - pos1) : 0;
  
  f.clear();
  //f.seekg(0, ios::beg);
  f.seekg(pos1);
  
  x.set_size(N / sizeof(eT), 1);
  
  f.clear();
  f.read( reinterpret_cast<char *>(x.memptr()), std::streamsize(N) );
  
  return f.good();
  }



//! Load a matrix in text format (human readable),
//! with a header that indicates the matrix type as well as its dimensions
template<typename eT>
inline
bool
diskio::load_arma_ascii(Mat<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::ifstream f(name.c_str());
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_arma_ascii(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



//! Load a matrix in text format (human readable),
//! with a header that indicates the matrix type as well as its dimensions
template<typename eT>
inline
bool
diskio::load_arma_ascii(Mat<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::streampos pos = f.tellg();
    
  bool load_okay = true;
  
  std::string f_header;
  uword f_n_rows;
  uword f_n_cols;
  
  f >> f_header;
  f >> f_n_rows;
  f >> f_n_cols;
  
  if(f_header == diskio::gen_txt_header(x))
    {
    x.zeros(f_n_rows, f_n_cols);
    
    std::string       token;
    std::stringstream ss;
    
    for(uword row=0; row < x.n_rows; ++row)
      {
      for(uword col=0; col < x.n_cols; ++col)
        {
        f >> token;
        
        ss.clear();
        ss.str(token);
        
        eT val = eT(0);
        ss >> val;
        
        if(ss.fail() == false)
          {
          x.at(row,col) = val;
          }
        else
          {
          diskio::convert_naninf( x.at(row,col), token );
          }
        }
      }
    
    load_okay = f.good();
    }
  else
    {
    load_okay = false;
    err_msg = "incorrect header in ";
    }
  
  
  // allow automatic conversion of u32/s32 matrices into u64/s64 matrices
  
  if(load_okay == false)
    {
    if( (sizeof(eT) == 8) && is_same_type<uword,eT>::yes )
      {
      Mat<u32>    tmp;
      std::string junk;
      
      f.clear();
      f.seekg(pos);
      
      load_okay = diskio::load_arma_ascii(tmp, f, junk);
      
      if(load_okay)  { x = conv_to< Mat<eT> >::from(tmp); }
      }
    else
    if( (sizeof(eT) == 8) && is_same_type<sword,eT>::yes )
      {
      Mat<s32>    tmp;
      std::string junk;
      
      f.clear();
      f.seekg(pos);
      
      load_okay = diskio::load_arma_ascii(tmp, f, junk);
      
      if(load_okay)  { x = conv_to< Mat<eT> >::from(tmp); }
      }
    }
  
  return load_okay;
  }



//! Load a matrix in CSV text format (human readable)
template<typename eT>
inline
bool
diskio::load_csv_ascii(Mat<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::fstream f;
  f.open(name.c_str(), std::fstream::in);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_csv_ascii(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



//! Load a matrix in CSV text format (human readable)
template<typename eT>
inline
bool
diskio::load_csv_ascii(Mat<eT>& x, std::istream& f, std::string&)
  {
  arma_extra_debug_sigprint();
  
  bool load_okay = f.good();
  
  f.clear();
  const std::fstream::pos_type pos1 = f.tellg();
  
  //
  // work out the size
  
  uword f_n_rows = 0;
  uword f_n_cols = 0;
  
  std::string line_string;
  std::string token;
  
  std::stringstream line_stream;
  
  while( (f.good() == true) && (load_okay == true) )
    {
    std::getline(f, line_string);
    
    if(line_string.size() == 0)
      {
      break;
      }
    
    line_stream.clear();
    line_stream.str(line_string);
    
    uword line_n_cols = 0;
    
    while(line_stream.good() == true)
      {
      std::getline(line_stream, token, ',');
      ++line_n_cols;
      }
    
    if(f_n_cols < line_n_cols)
      {
      f_n_cols = line_n_cols;
      }
    
    ++f_n_rows;
    }
  
  f.clear();
  f.seekg(pos1);
  
  x.zeros(f_n_rows, f_n_cols);
  
  uword row = 0;
  
  std::stringstream ss;
  
  while(f.good() == true)
    {
    std::getline(f, line_string);
    
    if(line_string.size() == 0)
      {
      break;
      }
    
    line_stream.clear();
    line_stream.str(line_string);
    
    uword col = 0;
    
    while(line_stream.good() == true)
      {
      std::getline(line_stream, token, ',');
      
      ss.clear();
      ss.str(token);
      
      eT val = eT(0);
      ss >> val;
      
      if(ss.fail() == false)
        {
        x.at(row,col) = val;
        }
      else
        {
        diskio::convert_naninf( x.at(row,col), token );
        }
      
      ++col;
      }
    
    ++row;
    }
  
  return load_okay;
  }



//! Load a matrix in binary format,
//! with a header that indicates the matrix type as well as its dimensions
template<typename eT>
inline
bool
diskio::load_arma_binary(Mat<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::ifstream f;
  f.open(name.c_str(), std::fstream::binary);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_arma_binary(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



template<typename eT>
inline
bool
diskio::load_arma_binary(Mat<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::streampos pos = f.tellg();
    
  bool load_okay = true;
  
  std::string f_header;
  uword f_n_rows;
  uword f_n_cols;
  
  f >> f_header;
  f >> f_n_rows;
  f >> f_n_cols;
  
  if(f_header == diskio::gen_bin_header(x))
    {
    //f.seekg(1, ios::cur);  // NOTE: this may not be portable, as on a Windows machine a newline could be two characters
    f.get();
    
    x.set_size(f_n_rows,f_n_cols);
    f.read( reinterpret_cast<char *>(x.memptr()), std::streamsize(x.n_elem*sizeof(eT)) );
    
    load_okay = f.good();
    }
  else
    {
    load_okay = false;
    err_msg = "incorrect header in ";
    }
  
  
  // allow automatic conversion of u32/s32 matrices into u64/s64 matrices
  
  if(load_okay == false)
    {
    if( (sizeof(eT) == 8) && is_same_type<uword,eT>::yes )
      {
      Mat<u32>    tmp;
      std::string junk;
      
      f.clear();
      f.seekg(pos);
      
      load_okay = diskio::load_arma_binary(tmp, f, junk);
      
      if(load_okay)  { x = conv_to< Mat<eT> >::from(tmp); }
      }
    else
    if( (sizeof(eT) == 8) && is_same_type<sword,eT>::yes )
      {
      Mat<s32>    tmp;
      std::string junk;
      
      f.clear();
      f.seekg(pos);
      
      load_okay = diskio::load_arma_binary(tmp, f, junk);
      
      if(load_okay)  { x = conv_to< Mat<eT> >::from(tmp); }
      }
    }
  
  return load_okay;
  }



inline
void
diskio::pnm_skip_comments(std::istream& f)
  {
  while( isspace(f.peek()) )
    {
    while( isspace(f.peek()) )
      {
      f.get();
      }
  
    if(f.peek() == '#')
      {
      while( (f.peek() != '\r') && (f.peek()!='\n') )
        {
        f.get();
        }
      }
    }
  }



//! Load a PGM greyscale image as a matrix
template<typename eT>
inline
bool
diskio::load_pgm_binary(Mat<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::fstream f;
  f.open(name.c_str(), std::fstream::in | std::fstream::binary);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_pgm_binary(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



//! Load a PGM greyscale image as a matrix
template<typename eT>
inline
bool
diskio::load_pgm_binary(Mat<eT>& x, std::istream& f, std::string& err_msg)
  {
  bool load_okay = true;
  
  std::string f_header;
  f >> f_header;
  
  if(f_header == "P5")
    {
    uword f_n_rows = 0;
    uword f_n_cols = 0;
    int f_maxval = 0;
  
    diskio::pnm_skip_comments(f);
  
    f >> f_n_cols;
    diskio::pnm_skip_comments(f);
  
    f >> f_n_rows;
    diskio::pnm_skip_comments(f);
  
    f >> f_maxval;
    f.get();
    
    if( (f_maxval > 0) || (f_maxval <= 65535) )
      {
      x.set_size(f_n_rows,f_n_cols);
      
      if(f_maxval <= 255)
        {
        const uword n_elem = f_n_cols*f_n_rows;
        podarray<u8> tmp(n_elem);
        
        f.read( reinterpret_cast<char*>(tmp.memptr()), std::streamsize(n_elem) );
        
        uword i = 0;
        
        //cout << "f_n_cols = " << f_n_cols << endl;
        //cout << "f_n_rows = " << f_n_rows << endl;
        
        
        for(uword row=0; row < f_n_rows; ++row)
          {
          for(uword col=0; col < f_n_cols; ++col)
            {
            x.at(row,col) = eT(tmp[i]);
            ++i;
            }
          }
          
        }
      else
        {
        const uword n_elem = f_n_cols*f_n_rows;
        podarray<u16> tmp(n_elem);
        
        f.read( reinterpret_cast<char *>(tmp.memptr()), std::streamsize(n_elem*2) );
        
        uword i = 0;
        
        for(uword row=0; row < f_n_rows; ++row)
          {
          for(uword col=0; col < f_n_cols; ++col)
            {
            x.at(row,col) = eT(tmp[i]);
            ++i;
            }
          }
        
        }
      
      }
    else
      {
      load_okay = false;
      err_msg = "currently no code available to handle loading ";
      }
    
    if(f.good() == false)
      {
      load_okay = false;
      }
    }
  else
    {
    load_okay = false;
    err_msg = "unsupported header in ";
    }
  
  return load_okay;
  }



//! Load a PGM greyscale image as a matrix
template<typename T>
inline
bool
diskio::load_pgm_binary(Mat< std::complex<T> >& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  uchar_mat tmp;
  const bool load_okay = diskio::load_pgm_binary(tmp, name, err_msg);
  
  x = conv_to< Mat< std::complex<T> > >::from(tmp);
  
  return load_okay;
  }



//! Load a PGM greyscale image as a matrix
template<typename T>
inline
bool
diskio::load_pgm_binary(Mat< std::complex<T> >& x, std::istream& is, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  uchar_mat tmp;
  const bool load_okay = diskio::load_pgm_binary(tmp, is, err_msg);
  
  x = conv_to< Mat< std::complex<T> > >::from(tmp);
  
  return load_okay;
  }



//! Load a HDF5 file as a matrix
template<typename eT>
inline
bool
diskio::load_hdf5_binary(Mat<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_HDF5)
    {

    // These may be necessary to store the error handler (if we need to).
    herr_t (*old_func)(hid_t, void*);
    void *old_client_data;

    #if !defined(ARMA_PRINT_HDF5_ERRORS)
      {
      // Save old error handler.
      arma_H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);

      // Disable annoying HDF5 error messages.
      arma_H5Eset_auto(H5E_DEFAULT, NULL, NULL);
      }
    #endif

    bool load_okay = false;
    
    hid_t fid = arma_H5Fopen(name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    
    if(fid >= 0)
      {
      // MATLAB HDF5 dataset names are user-specified;
      // Octave tends to store the datasets in a group, with the actual dataset being referred to as "value".
      // So we will search for "dataset" and "value", and if those are not found we will take the first dataset we do find.
      std::vector<std::string> searchNames;
      searchNames.push_back("dataset");
      searchNames.push_back("value");
      
      hid_t dataset = hdf5_misc::search_hdf5_file(searchNames, fid, 2, false);

      if(dataset >= 0)
        {
        hid_t filespace = arma_H5Dget_space(dataset);
        
        // This must be <= 2 due to our search rules.
        const int ndims = arma_H5Sget_simple_extent_ndims(filespace);
        
        hsize_t dims[2];
        const herr_t query_status = arma_H5Sget_simple_extent_dims(filespace, dims, NULL);
        
        // arma_check(query_status < 0, "Mat::load(): cannot get size of HDF5 dataset");
        if(query_status < 0)
          {
          err_msg = "cannot get size of HDF5 dataset in ";
          
          arma_H5Sclose(filespace);
          arma_H5Dclose(dataset);
          arma_H5Fclose(fid);
    
          #if !defined(ARMA_PRINT_HDF5_ERRORS)
            {
            // Restore HDF5 error handler.
            arma_H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);
            }
          #endif
          
          return false;
          }
        
        if(ndims == 1) { dims[1] = 1; }  // Vector case; fake second dimension (one column).
        
        x.set_size(dims[1], dims[0]);
        
        // Now we have to see what type is stored to figure out how to load it.
        hid_t datatype = arma_H5Dget_type(dataset);
        hid_t mat_type = hdf5_misc::get_hdf5_type<eT>();
        
        // If these are the same type, it is simple.
        if(arma_H5Tequal(datatype, mat_type) > 0)
          {
          // Load directly; H5S_ALL used so that we load the entire dataset.
          hid_t read_status = arma_H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, void_ptr(x.memptr()));
          
          if(read_status >= 0) { load_okay = true; }
          }
        else
          {
          // Load into another array and convert its type accordingly.
          hid_t read_status = hdf5_misc::load_and_convert_hdf5(x.memptr(), dataset, datatype, x.n_elem);
          
          if(read_status >= 0) { load_okay = true; }
          }
        
        // Now clean up.
        arma_H5Tclose(datatype);
        arma_H5Tclose(mat_type);
        arma_H5Sclose(filespace);
        }
      
      arma_H5Dclose(dataset);
    
      arma_H5Fclose(fid);

      if(load_okay == false)
        {
        err_msg = "unsupported or incorrect HDF5 data in ";
        }
      }
    else
      {
      err_msg = "cannot open file ";
      }

    #if !defined(ARMA_PRINT_HDF5_ERRORS)
      {
      // Restore HDF5 error handler.
      arma_H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);
      }
    #endif

    return load_okay;
    }
  #else
    {
    arma_ignore(x);
    arma_ignore(name);
    arma_ignore(err_msg);

    arma_stop("Mat::load(): use of HDF5 needs to be enabled");

    return false;
    }
  #endif
  }



//! Try to load a matrix by automatically determining its type
template<typename eT>
inline
bool
diskio::load_auto_detect(Mat<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_HDF5)
    // We're currently using the C bindings for the HDF5 library, which don't support C++ streams
    if( arma_H5Fis_hdf5(name.c_str()) ) { return load_hdf5_binary(x, name, err_msg); }
  #endif

  std::fstream f;
  f.open(name.c_str(), std::fstream::in | std::fstream::binary);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_auto_detect(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



//! Try to load a matrix by automatically determining its type
template<typename eT>
inline
bool
diskio::load_auto_detect(Mat<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  static const std::string ARMA_MAT_TXT = "ARMA_MAT_TXT";
  static const std::string ARMA_MAT_BIN = "ARMA_MAT_BIN";
  static const std::string           P5 = "P5";
  
  podarray<char> raw_header( uword(ARMA_MAT_TXT.length()) + 1);
  
  std::streampos pos = f.tellg();
    
  f.read( raw_header.memptr(), std::streamsize(ARMA_MAT_TXT.length()) );
  raw_header[uword(ARMA_MAT_TXT.length())] = '\0';
  
  f.clear();
  f.seekg(pos);
  
  const std::string header = raw_header.mem;
  
  if(ARMA_MAT_TXT == header.substr(0,ARMA_MAT_TXT.length()))
    {
    return load_arma_ascii(x, f, err_msg);
    }
  else
  if(ARMA_MAT_BIN == header.substr(0,ARMA_MAT_BIN.length()))
    {
    return load_arma_binary(x, f, err_msg);
    }
  else
  if(P5 == header.substr(0,P5.length()))
    {
    return load_pgm_binary(x, f, err_msg);
    }
  else
    {
    const file_type ft = guess_file_type(f);
    
    switch(ft)
      {
      case csv_ascii:
        return load_csv_ascii(x, f, err_msg);
        break;
      
      case raw_binary:
        return load_raw_binary(x, f, err_msg);
        break;
        
      case raw_ascii:
        return load_raw_ascii(x, f, err_msg);
        break;
      
      default:
        err_msg = "unknown data in ";
        return false;
      }
    }
  
  return false;
  }



//
// sparse matrices
//



//! Save a matrix in ASCII coord format
template<typename eT>
inline
bool
diskio::save_coord_ascii(const SpMat<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();

  const std::string tmp_name = diskio::gen_tmp_name(final_name);

  std::ofstream f(tmp_name.c_str());

  bool save_okay = f.is_open();

  if(save_okay == true)
    {
    save_okay = diskio::save_coord_ascii(x, f);

    f.flush();
    f.close();

    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }

  return save_okay;
  }



//! Save a matrix in ASCII coord format
template<typename eT>
inline
bool
diskio::save_coord_ascii(const SpMat<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  const ios::fmtflags orig_flags = f.flags();
  
  typename SpMat<eT>::const_iterator iter     = x.begin();
  typename SpMat<eT>::const_iterator iter_end = x.end();
  
  for(; iter != iter_end; ++iter)
    {
    f.setf(ios::fixed);
    
    f << iter.row() << ' ' << iter.col() << ' ';
    
    if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )
      {
      f.setf(ios::scientific);
      f.precision(12);
      }
    
    f << (*iter) << '\n';
    }
  
  
  // make sure it's possible to figure out the matrix size later
  if( (x.n_rows > 0) && (x.n_cols > 0) )
    {
    const uword max_row = (x.n_rows > 0) ? x.n_rows-1 : 0;
    const uword max_col = (x.n_cols > 0) ? x.n_cols-1 : 0;
    
    if( x.at(max_row, max_col) == eT(0) )
      {
      f.setf(ios::fixed);
      
      f << max_row << ' ' << max_col << " 0\n";
      }
    }
  
  const bool save_okay = f.good();
  
  f.flags(orig_flags);
  
  return save_okay;
  }



//! Save a matrix in ASCII coord format (complex numbers)
template<typename T>
inline
bool
diskio::save_coord_ascii(const SpMat< std::complex<T> >& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  const ios::fmtflags orig_flags = f.flags();
  
  typedef typename std::complex<T> eT;
  
  typename SpMat<eT>::const_iterator iter     = x.begin();
  typename SpMat<eT>::const_iterator iter_end = x.end();
  
  for(; iter != iter_end; ++iter)
    {
    f.setf(ios::fixed);
    
    f << iter.row() << ' ' << iter.col() << ' ';
    
    if( (is_float<T>::value == true) || (is_double<T>::value == true) )
      {
      f.setf(ios::scientific);
      f.precision(12);
      }
    
    const eT val = (*iter);
    
    f << val.real() << ' ' << val.imag() << '\n';
    }
  
  // make sure it's possible to figure out the matrix size later
  if( (x.n_rows > 0) && (x.n_cols > 0) )
    {
    const uword max_row = (x.n_rows > 0) ? x.n_rows-1 : 0;
    const uword max_col = (x.n_cols > 0) ? x.n_cols-1 : 0;
    
    if( x.at(max_row, max_col) == eT(0) )
      {
      f.setf(ios::fixed);
      
      f << max_row << ' ' << max_col << " 0 0\n";
      }
    }
  
  const bool save_okay = f.good();
  
  f.flags(orig_flags);
  
  return save_okay;
  }



//! Save a matrix in binary format,
//! with a header that stores the matrix type as well as its dimensions
template<typename eT>
inline
bool
diskio::save_arma_binary(const SpMat<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();

  const std::string tmp_name = diskio::gen_tmp_name(final_name);

  std::ofstream f(tmp_name.c_str(), std::fstream::binary);

  bool save_okay = f.is_open();

  if(save_okay == true)
    {
    save_okay = diskio::save_arma_binary(x, f);

    f.flush();
    f.close();

    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }

  return save_okay;
  }



//! Save a matrix in binary format,
//! with a header that stores the matrix type as well as its dimensions
template<typename eT>
inline
bool
diskio::save_arma_binary(const SpMat<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  f << diskio::gen_bin_header(x) << '\n';
  f << x.n_rows << ' ' << x.n_cols << ' ' << x.n_nonzero << '\n';
  
  f.write( reinterpret_cast<const char*>(x.values),      std::streamsize(x.n_nonzero*sizeof(eT))     );
  f.write( reinterpret_cast<const char*>(x.row_indices), std::streamsize(x.n_nonzero*sizeof(uword))  );
  f.write( reinterpret_cast<const char*>(x.col_ptrs),    std::streamsize((x.n_cols+1)*sizeof(uword)) );
  
  return f.good();
  }



template<typename eT>
inline
bool
diskio::load_coord_ascii(SpMat<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::fstream f;
  f.open(name.c_str(), std::fstream::in | std::fstream::binary);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_coord_ascii(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



template<typename eT>
inline
bool
diskio::load_coord_ascii(SpMat<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  arma_ignore(err_msg);
  
  bool load_okay = f.good();
  
  f.clear();
  const std::fstream::pos_type pos1 = f.tellg();
  
  //
  // work out the size
  
  uword f_n_rows = 0;
  uword f_n_cols = 0;
  uword f_n_nz   = 0;
  
  bool size_found = false;
  
  std::string       line_string;
  std::string       token;
  
  std::stringstream line_stream;
  std::stringstream ss;
  
  uword last_line_row = 0;
  uword last_line_col = 0;
  
  bool first_line   = true;
  bool weird_format = false;
  
  
  while( (f.good() == true) && (load_okay == true) )
    {
    std::getline(f, line_string);
    
    if(line_string.size() == 0)
      {
      break;
      }
    
    line_stream.clear();
    line_stream.str(line_string);
    
    uword line_row = 0;
    uword line_col = 0;
    
    // a valid line in co-ord format has at least 2 entries
    
    line_stream >> line_row;
    
    if(line_stream.good() == false)
      {
      load_okay = false;
      break;
      }
    
    line_stream >> line_col;
    
    size_found = true;
    
    if(f_n_rows < line_row)  f_n_rows = line_row;
    if(f_n_cols < line_col)  f_n_cols = line_col;
    
    if(first_line == true)
      {
      first_line = false;
      }
    else
      {
      if( (line_col < last_line_col) || ((line_row <= last_line_row) && (line_col <= last_line_col)) )
        {
        weird_format = true;
        }
      }
    
    last_line_row = line_row;
    last_line_col = line_col;
    
    
    if(line_stream.good() == true)
      {
      eT final_val = eT(0);
      
      line_stream >> token;
      
      if(line_stream.fail() == false)
        {
        eT val = eT(0);
        
        ss.clear();
        ss.str(token);
        
        ss >> val;
        
        if(ss.fail() == false)
          {
          final_val = val;
          }
        else
          {
          val = eT(0);
          
          const bool success = diskio::convert_naninf( val, token );
          
          if(success == true)
            {
            final_val = val;
            }
          }
        }
      
      if(final_val != eT(0))
        {
        ++f_n_nz;
        }
      }
    }
  
  
  if(size_found == true)
    {
    // take into account that indices start at 0
    f_n_rows++;
    f_n_cols++;
    }
  
  
  if(load_okay == true)
    {
    f.clear();
    f.seekg(pos1);
    
    x.set_size(f_n_rows, f_n_cols);
    
    if(weird_format == false)
      {
      x.mem_resize(f_n_nz);
      }
    
    uword pos = 0;
    
    while(f.good() == true)
      {
      std::getline(f, line_string);
      
      if(line_string.size() == 0)
        {
        break;
        }
      
      line_stream.clear();
      line_stream.str(line_string);
      
      uword line_row = 0;
      uword line_col = 0;
      
      line_stream >> line_row;
      line_stream >> line_col;
      
      eT final_val = eT(0);
      
      line_stream >> token;
      
      if(line_stream.fail() == false)
        {
        eT val = eT(0);
        
        ss.clear();
        ss.str(token);
        
        ss >> val;
        
        if(ss.fail() == false)
          {
          final_val = val;
          }
        else
          {
          val = eT(0);
          
          const bool success = diskio::convert_naninf( val, token );
          
          if(success == true)
            {
            final_val = val;
            }
          }
        }
      
      
      if(final_val != eT(0))
        {
        if(weird_format == false)
          {
          access::rw(x.row_indices[pos]) = line_row;
          access::rw(x.values[pos])      = final_val;
          ++access::rw(x.col_ptrs[line_col + 1]);
          
          ++pos;
          }
        else
          {
          x.at(line_row,line_col) = final_val;
          }
        }
      }
    
    if(weird_format == false)
      {
      for(uword c = 1; c <= f_n_cols; ++c)
        {
        access::rw(x.col_ptrs[c]) += x.col_ptrs[c - 1];
        }
      }
    }
  
  return load_okay;
  }



template<typename T>
inline
bool
diskio::load_coord_ascii(SpMat< std::complex<T> >& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  arma_ignore(err_msg);
  
  bool load_okay = f.good();
  
  f.clear();
  const std::fstream::pos_type pos1 = f.tellg();
  
  //
  // work out the size
  
  uword f_n_rows = 0;
  uword f_n_cols = 0;
  uword f_n_nz   = 0;
  
  bool size_found = false;
  
  std::string line_string;
  std::string token_real;
  std::string token_imag;
  
  std::stringstream line_stream;
  std::stringstream ss;
  
  uword last_line_row = 0;
  uword last_line_col = 0;
  
  bool first_line   = true;
  bool weird_format = false;
  
  while( (f.good() == true) && (load_okay == true) )
    {
    std::getline(f, line_string);
    
    if(line_string.size() == 0)
      {
      break;
      }
    
    line_stream.clear();
    line_stream.str(line_string);
    
    uword line_row = 0;
    uword line_col = 0;
    
    // a valid line in co-ord format has at least 2 entries
    
    line_stream >> line_row;
    
    if(line_stream.good() == false)
      {
      load_okay = false;
      break;
      }
    
    line_stream >> line_col;
    
    size_found = true;
    
    if(f_n_rows < line_row)  f_n_rows = line_row;
    if(f_n_cols < line_col)  f_n_cols = line_col;
    
    
    if(first_line == true)
      {
      first_line = false;
      }
    else
      {
      if( (line_col < last_line_col) || ((line_row <= last_line_row) && (line_col <= last_line_col)) )
        {
        weird_format = true;
        }
      }
    
    last_line_row = line_row;
    last_line_col = line_col;
    
    
    if(line_stream.good() == true)
      {
      T final_val_real = T(0);
      T final_val_imag = T(0);
      
      
      line_stream >> token_real;
      
      if(line_stream.fail() == false)
        {
        T val_real = T(0);
        
        ss.clear();
        ss.str(token_real);
        
        ss >> val_real;
        
        if(ss.fail() == false)
          {
          final_val_real = val_real;
          }
        else
          {
          val_real = T(0);
          
          const bool success = diskio::convert_naninf( val_real, token_real );
          
          if(success == true)
            {
            final_val_real = val_real;
            }
          }
        }
      
      
      line_stream >> token_imag;
      
      if(line_stream.fail() == false)
        {
        T val_imag = T(0);
        
        ss.clear();
        ss.str(token_imag);
        
        ss >> val_imag;
        
        if(ss.fail() == false)
          {
          final_val_imag = val_imag;
          }
        else
          {
          val_imag = T(0);
          
          const bool success = diskio::convert_naninf( val_imag, token_imag );
          
          if(success == true)
            {
            final_val_imag = val_imag;
            }
          }
        }
      
      
      if( (final_val_real != T(0)) || (final_val_imag != T(0)) )
        {
        ++f_n_nz;
        }
      }
    }
  
  
  if(size_found == true)
    {
    // take into account that indices start at 0
    f_n_rows++;
    f_n_cols++;
    }
  
  
  if(load_okay == true)
    {
    f.clear();
    f.seekg(pos1);
    
    x.set_size(f_n_rows, f_n_cols);
    
    if(weird_format == false)
      {
      x.mem_resize(f_n_nz);
      }
    
    uword pos = 0;
    
    while(f.good() == true)
      {
      std::getline(f, line_string);
      
      if(line_string.size() == 0)
        {
        break;
        }
      
      line_stream.clear();
      line_stream.str(line_string);
      
      uword line_row = 0;
      uword line_col = 0;
      
      line_stream >> line_row;
      line_stream >> line_col;
      
      T final_val_real = T(0);
      T final_val_imag = T(0);
      
      
      line_stream >> token_real;
      
      if(line_stream.fail() == false)
        {
        T val_real = T(0);
        
        ss.clear();
        ss.str(token_real);
        
        ss >> val_real;
        
        if(ss.fail() == false)
          {
          final_val_real = val_real;
          }
        else
          {
          val_real = T(0);
          
          const bool success = diskio::convert_naninf( val_real, token_real );
          
          if(success == true)
            {
            final_val_real = val_real;
            }
          }
        }
      
      
      line_stream >> token_imag;
      
      if(line_stream.fail() == false)
        {
        T val_imag = T(0);
        
        ss.clear();
        ss.str(token_imag);
        
        ss >> val_imag;
        
        if(ss.fail() == false)
          {
          final_val_imag = val_imag;
          }
        else
          {
          val_imag = T(0);
          
          const bool success = diskio::convert_naninf( val_imag, token_imag );
          
          if(success == true)
            {
            final_val_imag = val_imag;
            }
          }
        }
      
      
      if( (final_val_real != T(0)) || (final_val_imag != T(0)) )
        {
        if(weird_format == false)
          {
          access::rw(x.row_indices[pos]) = line_row;
          access::rw(x.values[pos])      = std::complex<T>(final_val_real, final_val_imag);
          ++access::rw(x.col_ptrs[line_col + 1]);
          
          ++pos;
          }
        else
          {
          x.at(line_row,line_col) = std::complex<T>(final_val_real, final_val_imag);
          }
        }
      }
    
    
    if(weird_format == false)
      {
      for(uword c = 1; c <= f_n_cols; ++c)
        {
        access::rw(x.col_ptrs[c]) += x.col_ptrs[c - 1];
        }
      }
    }
  
  return load_okay;
  }



//! Load a matrix in binary format,
//! with a header that indicates the matrix type as well as its dimensions
template<typename eT>
inline
bool
diskio::load_arma_binary(SpMat<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();

  std::ifstream f;
  f.open(name.c_str(), std::fstream::binary);

  bool load_okay = f.is_open();

  if(load_okay == true)
    {
    load_okay = diskio::load_arma_binary(x, f, err_msg);
    f.close();
    }

  return load_okay;
  }



template<typename eT>
inline
bool
diskio::load_arma_binary(SpMat<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  bool load_okay = true;
  
  std::string f_header;
  
  f >> f_header;
  
  if(f_header == diskio::gen_bin_header(x))
    {
    uword f_n_rows;
    uword f_n_cols;
    uword f_n_nz;
    
    f >> f_n_rows;
    f >> f_n_cols;
    f >> f_n_nz;
    
    //f.seekg(1, ios::cur);  // NOTE: this may not be portable, as on a Windows machine a newline could be two characters
    f.get();
    
    x.set_size(f_n_rows, f_n_cols);
    
    x.mem_resize(f_n_nz);
    
    f.read( reinterpret_cast<char*>(access::rwp(x.values)),      std::streamsize(x.n_nonzero*sizeof(eT))     );
    
    std::streampos pos = f.tellg();
    
    f.read( reinterpret_cast<char*>(access::rwp(x.row_indices)), std::streamsize(x.n_nonzero*sizeof(uword))  );
    f.read( reinterpret_cast<char*>(access::rwp(x.col_ptrs)),    std::streamsize((x.n_cols+1)*sizeof(uword)) );
    
    bool check1 = true;  for(uword i=0; i < x.n_nonzero; ++i)  { if(x.values[i] == eT(0))  { check1 = false; break; } }
    bool check2 = true;  for(uword i=0; i < x.n_cols;    ++i)  { if(x.col_ptrs[i+1] < x.col_ptrs[i])  { check2 = false; break; } }
    bool check3 = (x.col_ptrs[x.n_cols] == x.n_nonzero);
    
    if((check1 == true) && ((check2 == false) || (check3 == false)))
      {
      if(sizeof(uword) == 8)
        {
        arma_extra_debug_print("detected inconsistent data while loading; re-reading integer parts as u32");
        
        // inconstency could be due to a different uword size used during saving,
        // so try loading the row_indices and col_ptrs under the assumption of 32 bit unsigned integers
        
        f.clear();
        f.seekg(pos);
        
        podarray<u32> tmp_a(x.n_nonzero );  tmp_a.zeros();
        podarray<u32> tmp_b(x.n_cols + 1);  tmp_b.zeros();
        
        f.read( reinterpret_cast<char*>(tmp_a.memptr()), std::streamsize( x.n_nonzero   * sizeof(u32)) );
        f.read( reinterpret_cast<char*>(tmp_b.memptr()), std::streamsize((x.n_cols + 1) * sizeof(u32)) );
        
        check2 = true;  for(uword i=0; i < x.n_cols; ++i)  { if(tmp_b[i+1] < tmp_b[i])  { check2 = false; break; } }
        check3 = (tmp_b[x.n_cols] == x.n_nonzero);
        
        load_okay = f.good();
        
        if( load_okay && (check2 == true) && (check3 == true) )
          {
          arma_extra_debug_print("reading integer parts as u32 succeeded");
          
          arrayops::convert(access::rwp(x.row_indices), tmp_a.memptr(), x.n_nonzero );
          arrayops::convert(access::rwp(x.col_ptrs),    tmp_b.memptr(), x.n_cols + 1);
          }
        else
          {
          arma_extra_debug_print("reading integer parts as u32 failed");
          }
        }
      }
    
    if((check1 == false) || (check2 == false) || (check3 == false))
      {
      load_okay = false;
      err_msg = "inconsistent data in ";
      }
    else
      {
      load_okay = f.good();
      }
    }
  else
    {
    load_okay = false;
    err_msg = "incorrect header in ";
    }

  return load_okay;
  }



// cubes



//! Save a cube as raw text (no header, human readable).
template<typename eT>
inline
bool
diskio::save_raw_ascii(const Cube<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::fstream f(tmp_name.c_str(), std::fstream::out);
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = save_raw_ascii(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



//! Save a cube as raw text (no header, human readable).
template<typename eT>
inline
bool
diskio::save_raw_ascii(const Cube<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  uword cell_width;
  
  // TODO: need sane values for complex numbers
  
  if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )
    {
    f.setf(ios::scientific);
    f.precision(12);
    cell_width = 20;
    }
  
  for(uword slice=0; slice < x.n_slices; ++slice)
    {
    for(uword row=0; row < x.n_rows; ++row)
      {
      for(uword col=0; col < x.n_cols; ++col)
        {
        f.put(' ');
        
        if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )
          {
          f.width(cell_width);
          }
        
        arma_ostream::print_elem(f, x.at(row,col,slice), false);
        }
        
      f.put('\n');
      }
    }
  
  return f.good();
  }



//! Save a cube as raw binary (no header)
template<typename eT>
inline
bool
diskio::save_raw_binary(const Cube<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::ofstream f(tmp_name.c_str(), std::fstream::binary);
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = diskio::save_raw_binary(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



template<typename eT>
inline
bool
diskio::save_raw_binary(const Cube<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  f.write( reinterpret_cast<const char*>(x.mem), std::streamsize(x.n_elem*sizeof(eT)) );
  
  return f.good();
  }



//! Save a cube in text format (human readable),
//! with a header that indicates the cube type as well as its dimensions
template<typename eT>
inline
bool
diskio::save_arma_ascii(const Cube<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::ofstream f(tmp_name.c_str());
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = diskio::save_arma_ascii(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



//! Save a cube in text format (human readable),
//! with a header that indicates the cube type as well as its dimensions
template<typename eT>
inline
bool
diskio::save_arma_ascii(const Cube<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  const ios::fmtflags orig_flags = f.flags();
  
  f << diskio::gen_txt_header(x) << '\n';
  f << x.n_rows << ' ' << x.n_cols << ' ' << x.n_slices << '\n';
  
  uword cell_width;
  
  // TODO: need sane values for complex numbers
  
  if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )
    {
    f.setf(ios::scientific);
    f.precision(12);
    cell_width = 20;
    }
    
  for(uword slice=0; slice < x.n_slices; ++slice)
    {
    for(uword row=0; row < x.n_rows; ++row)
      {
      for(uword col=0; col < x.n_cols; ++col)
        {
        f.put(' ');
        
        if( (is_float<eT>::value == true) || (is_double<eT>::value == true) )        
          {
          f.width(cell_width);
          }
        
        arma_ostream::print_elem(f, x.at(row,col,slice), false);
        }
      
      f.put('\n');
      }
    }
  
  const bool save_okay = f.good();
  
  f.flags(orig_flags);
  
  return save_okay;
  }



//! Save a cube in binary format,
//! with a header that stores the cube type as well as its dimensions
template<typename eT>
inline
bool
diskio::save_arma_binary(const Cube<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::ofstream f(tmp_name.c_str(), std::fstream::binary);
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = diskio::save_arma_binary(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



//! Save a cube in binary format,
//! with a header that stores the cube type as well as its dimensions
template<typename eT>
inline
bool
diskio::save_arma_binary(const Cube<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  f << diskio::gen_bin_header(x) << '\n';
  f << x.n_rows << ' ' << x.n_cols << ' ' << x.n_slices << '\n';
  
  f.write( reinterpret_cast<const char*>(x.mem), std::streamsize(x.n_elem*sizeof(eT)) );
  
  return f.good();
  }



//! Save a cube as part of a HDF5 file
template<typename eT>
inline
bool
diskio::save_hdf5_binary(const Cube<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();

  #if defined(ARMA_USE_HDF5)
    {
    #if !defined(ARMA_PRINT_HDF5_ERRORS)
      {
      // Disable annoying HDF5 error messages.
      arma_H5Eset_auto(H5E_DEFAULT, NULL, NULL);
      }
    #endif

    bool save_okay = false;

    const std::string tmp_name = diskio::gen_tmp_name(final_name);

    // Set up the file according to HDF5's preferences
    hid_t file = arma_H5Fcreate(tmp_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // We need to create a dataset, datatype, and dataspace
    hsize_t dims[3];
    dims[2] = x.n_rows;
    dims[1] = x.n_cols;
    dims[0] = x.n_slices;

    hid_t dataspace = arma_H5Screate_simple(3, dims, NULL);   // treat the cube as a 3d array dataspace
    hid_t datatype  = hdf5_misc::get_hdf5_type<eT>();

    // If this returned something invalid, well, it's time to crash.
    arma_check(datatype == -1, "Cube::save(): unknown datatype for HDF5");

    // MATLAB forces the users to specify a name at save time for HDF5; Octave
    // will use the default of 'dataset' unless otherwise specified, so we will
    // use that.
    hid_t dataset = arma_H5Dcreate(file, "dataset", datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    herr_t status = arma_H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, x.mem);
    save_okay = (status >= 0);

    arma_H5Dclose(dataset);
    arma_H5Tclose(datatype);
    arma_H5Sclose(dataspace);
    arma_H5Fclose(file);

    if(save_okay == true) { save_okay = diskio::safe_rename(tmp_name, final_name); }

    return save_okay;
    }
  #else
    {
    arma_ignore(x);
    arma_ignore(final_name);

    arma_stop("Cube::save(): use of HDF5 needs to be enabled");

    return false;
    }
  #endif
  }



//! Load a cube as raw text (no header, human readable).
//! NOTE: this is much slower than reading a file with a header.
template<typename eT>
inline
bool
diskio::load_raw_ascii(Cube<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  Mat<eT> tmp;
  const bool load_okay = diskio::load_raw_ascii(tmp, name, err_msg);
  
  if(load_okay == true)
    {
    if(tmp.is_empty() == false)
      {
      x.set_size(tmp.n_rows, tmp.n_cols, 1);
      
      x.slice(0) = tmp;
      }
    else
      {
      x.reset();
      }
    }
  
  return load_okay;
  }



//! Load a cube as raw text (no header, human readable).
//! NOTE: this is much slower than reading a file with a header.
template<typename eT>
inline
bool
diskio::load_raw_ascii(Cube<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  Mat<eT> tmp;
  const bool load_okay = diskio::load_raw_ascii(tmp, f, err_msg);
  
  if(load_okay == true)
    {
    if(tmp.is_empty() == false)
      {
      x.set_size(tmp.n_rows, tmp.n_cols, 1);
      
      x.slice(0) = tmp;
      }
    else
      {
      x.reset();
      }
    }
  
  return load_okay;
  }



//! Load a cube in binary format (no header);
//! the cube is assumed to have one slice with one column
template<typename eT>
inline
bool
diskio::load_raw_binary(Cube<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::ifstream f;
  f.open(name.c_str(), std::fstream::binary);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_raw_binary(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



template<typename eT>
inline
bool
diskio::load_raw_binary(Cube<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  arma_ignore(err_msg);
  
  f.clear();
  const std::streampos pos1 = f.tellg();
  
  f.clear();
  f.seekg(0, ios::end);
  
  f.clear();
  const std::streampos pos2 = f.tellg();
  
  const uword N = ( (pos1 >= 0) && (pos2 >= 0) ) ? uword(pos2 - pos1) : 0;
  
  f.clear();
  //f.seekg(0, ios::beg);
  f.seekg(pos1);
  
  x.set_size(N / sizeof(eT), 1, 1);
  
  f.clear();
  f.read( reinterpret_cast<char *>(x.memptr()), std::streamsize(N) );
  
  return f.good();
  }



//! Load a cube in text format (human readable),
//! with a header that indicates the cube type as well as its dimensions
template<typename eT>
inline
bool
diskio::load_arma_ascii(Cube<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::ifstream f(name.c_str());
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_arma_ascii(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }
  


//! Load a cube in text format (human readable),
//! with a header that indicates the cube type as well as its dimensions
template<typename eT>
inline
bool
diskio::load_arma_ascii(Cube<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::streampos pos = f.tellg();
    
  bool load_okay = true;
  
  std::string f_header;
  uword f_n_rows;
  uword f_n_cols;
  uword f_n_slices;
  
  f >> f_header;
  f >> f_n_rows;
  f >> f_n_cols;
  f >> f_n_slices;
  
  if(f_header == diskio::gen_txt_header(x))
    {
    x.set_size(f_n_rows, f_n_cols, f_n_slices);

    for(uword slice=0; slice < x.n_slices; ++slice)
      {
      for(uword row=0; row < x.n_rows; ++row)
        {
        for(uword col=0; col < x.n_cols; ++col)
          {
          f >> x.at(row,col,slice);
          }
        }
      }
    
    load_okay = f.good();
    }
  else
    {
    load_okay = false;
    err_msg = "incorrect header in ";
    }
  
  
  // allow automatic conversion of u32/s32 cubes into u64/s64 cubes
  
  if(load_okay == false)
    {
    if( (sizeof(eT) == 8) && is_same_type<uword,eT>::yes )
      {
      Cube<u32>   tmp;
      std::string junk;
      
      f.clear();
      f.seekg(pos);
      
      load_okay = diskio::load_arma_ascii(tmp, f, junk);
      
      if(load_okay)  { x = conv_to< Cube<eT> >::from(tmp); }
      }
    else
    if( (sizeof(eT) == 8) && is_same_type<sword,eT>::yes )
      {
      Cube<s32>   tmp;
      std::string junk;
      
      f.clear();
      f.seekg(pos);
      
      load_okay = diskio::load_arma_ascii(tmp, f, junk);
      
      if(load_okay)  { x = conv_to< Cube<eT> >::from(tmp); }
      }
    }
  
  return load_okay;
  }



//! Load a cube in binary format,
//! with a header that indicates the cube type as well as its dimensions
template<typename eT>
inline
bool
diskio::load_arma_binary(Cube<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::ifstream f;
  f.open(name.c_str(), std::fstream::binary);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_arma_binary(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



template<typename eT>
inline
bool
diskio::load_arma_binary(Cube<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::streampos pos = f.tellg();
    
  bool load_okay = true;
  
  std::string f_header;
  uword f_n_rows;
  uword f_n_cols;
  uword f_n_slices;
  
  f >> f_header;
  f >> f_n_rows;
  f >> f_n_cols;
  f >> f_n_slices;
  
  if(f_header == diskio::gen_bin_header(x))
    {
    //f.seekg(1, ios::cur);  // NOTE: this may not be portable, as on a Windows machine a newline could be two characters
    f.get();
    
    x.set_size(f_n_rows, f_n_cols, f_n_slices);
    f.read( reinterpret_cast<char *>(x.memptr()), std::streamsize(x.n_elem*sizeof(eT)) );
    
    load_okay = f.good();
    }
  else
    {
    load_okay = false;
    err_msg = "incorrect header in ";
    }
  
  
  // allow automatic conversion of u32/s32 cubes into u64/s64 cubes
  
  if(load_okay == false)
    {
    if( (sizeof(eT) == 8) && is_same_type<uword,eT>::yes )
      {
      Cube<u32>   tmp;
      std::string junk;
      
      f.clear();
      f.seekg(pos);
      
      load_okay = diskio::load_arma_binary(tmp, f, junk);
      
      if(load_okay)  { x = conv_to< Cube<eT> >::from(tmp); }
      }
    else
    if( (sizeof(eT) == 8) && is_same_type<sword,eT>::yes )
      {
      Cube<s32>   tmp;
      std::string junk;
      
      f.clear();
      f.seekg(pos);
      
      load_okay = diskio::load_arma_binary(tmp, f, junk);
      
      if(load_okay)  { x = conv_to< Cube<eT> >::from(tmp); }
      }
    }
  
  return load_okay;
  }



//! Load a HDF5 file as a cube
template<typename eT>
inline
bool
diskio::load_hdf5_binary(Cube<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();

  #if defined(ARMA_USE_HDF5)
    {

    // These may be necessary to store the error handler (if we need to).
    herr_t (*old_func)(hid_t, void*);
    void *old_client_data;

    #if !defined(ARMA_PRINT_HDF5_ERRORS)
      {
      // Save old error handler.
      arma_H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);

      // Disable annoying HDF5 error messages.
      arma_H5Eset_auto(H5E_DEFAULT, NULL, NULL);
      }
    #endif

    bool load_okay = false;

    hid_t fid = arma_H5Fopen(name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    if(fid >= 0)
      {
      // MATLAB HDF5 dataset names are user-specified;
      // Octave tends to store the datasets in a group, with the actual dataset being referred to as "value".
      // So we will search for "dataset" and "value", and if those are not found we will take the first dataset we do find.
      std::vector<std::string> searchNames;
      searchNames.push_back("dataset");
      searchNames.push_back("value");

      hid_t dataset = hdf5_misc::search_hdf5_file(searchNames, fid, 3, false);

      if(dataset >= 0)
        {
        hid_t filespace = arma_H5Dget_space(dataset);

        // This must be <= 3 due to our search rules.
        const int ndims = arma_H5Sget_simple_extent_ndims(filespace);

        hsize_t dims[3];
        const herr_t query_status = arma_H5Sget_simple_extent_dims(filespace, dims, NULL);

        // arma_check(query_status < 0, "Cube::load(): cannot get size of HDF5 dataset");
        if(query_status < 0)
          {
          err_msg = "cannot get size of HDF5 dataset in ";

          arma_H5Sclose(filespace);
          arma_H5Dclose(dataset);
          arma_H5Fclose(fid);

          #if !defined(ARMA_PRINT_HDF5_ERRORS)
            {
            // Restore HDF5 error handler.
            arma_H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);
            }
          #endif

          return false;
          }

        if (ndims == 1) { dims[1] = 1; dims[2] = 1; }  // Vector case; one row/colum, several slices
        if (ndims == 2) { dims[2] = 1; } // Matrix case; one column, several rows/slices

        x.set_size(dims[2], dims[1], dims[0]);

        // Now we have to see what type is stored to figure out how to load it.
        hid_t datatype = arma_H5Dget_type(dataset);
        hid_t mat_type = hdf5_misc::get_hdf5_type<eT>();

        // If these are the same type, it is simple.
        if(arma_H5Tequal(datatype, mat_type) > 0)
          {
          // Load directly; H5S_ALL used so that we load the entire dataset.
          hid_t read_status = arma_H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, void_ptr(x.memptr()));

          if(read_status >= 0) { load_okay = true; }
          }
        else
          {
          // Load into another array and convert its type accordingly.
          hid_t read_status = hdf5_misc::load_and_convert_hdf5(x.memptr(), dataset, datatype, x.n_elem);

          if(read_status >= 0) { load_okay = true; }
          }

        // Now clean up.
        arma_H5Tclose(datatype);
        arma_H5Tclose(mat_type);
        arma_H5Sclose(filespace);
        }

      arma_H5Dclose(dataset);

      arma_H5Fclose(fid);

      if(load_okay == false)
        {
        err_msg = "unsupported or incorrect HDF5 data in ";
        }
      }
    else
      {
      err_msg = "cannot open file ";
      }

    #if !defined(ARMA_PRINT_HDF5_ERRORS)
      {
      // Restore HDF5 error handler.
      arma_H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);
      }
    #endif

    return load_okay;
    }
  #else
    {
    arma_ignore(x);
    arma_ignore(name);
    arma_ignore(err_msg);

    arma_stop("Cube::load(): use of HDF5 needs to be enabled");

    return false;
    }
  #endif
  }



//! Try to load a cube by automatically determining its type
template<typename eT>
inline
bool
diskio::load_auto_detect(Cube<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  #if defined(ARMA_USE_HDF5)
    // We're currently using the C bindings for the HDF5 library, which don't support C++ streams
    if( arma_H5Fis_hdf5(name.c_str()) ) { return load_hdf5_binary(x, name, err_msg); }
  #endif

  std::fstream f;
  f.open(name.c_str(), std::fstream::in | std::fstream::binary);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_auto_detect(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



//! Try to load a cube by automatically determining its type
template<typename eT>
inline
bool
diskio::load_auto_detect(Cube<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  static const std::string ARMA_CUB_TXT = "ARMA_CUB_TXT";
  static const std::string ARMA_CUB_BIN = "ARMA_CUB_BIN";
  static const std::string           P6 = "P6";
  
  podarray<char> raw_header(uword(ARMA_CUB_TXT.length()) + 1);
  
  std::streampos pos = f.tellg();
  
  f.read( raw_header.memptr(), std::streamsize(ARMA_CUB_TXT.length()) );
  raw_header[uword(ARMA_CUB_TXT.length())] = '\0';
  
  f.clear();
  f.seekg(pos);
  
  const std::string header = raw_header.mem;
  
  if(ARMA_CUB_TXT == header.substr(0, ARMA_CUB_TXT.length()))
    {
    return load_arma_ascii(x, f, err_msg);
    }
  else
  if(ARMA_CUB_BIN == header.substr(0, ARMA_CUB_BIN.length()))
    {
    return load_arma_binary(x, f, err_msg);
    }
  else
  if(P6 == header.substr(0, P6.length()))
    {
    return load_ppm_binary(x, f, err_msg);
    }
  else
    {
    const file_type ft = guess_file_type(f);
    
    switch(ft)
      {
      // case csv_ascii:
      //   return load_csv_ascii(x, f, err_msg);
      //   break;
      
      case raw_binary:
        return load_raw_binary(x, f, err_msg);
        break;
        
      case raw_ascii:
        return load_raw_ascii(x, f, err_msg);
        break;
        
      default:
        err_msg = "unknown data in ";
        return false;
      }
    }
  
  return false;
  }





// fields



template<typename T1>
inline
bool
diskio::save_arma_binary(const field<T1>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::ofstream f( tmp_name.c_str(), std::fstream::binary );
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = diskio::save_arma_binary(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



template<typename T1>
inline
bool
diskio::save_arma_binary(const field<T1>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( (is_Mat<T1>::value == false) && (is_Cube<T1>::value == false) ));
  
  if(x.n_slices <= 1)
    {
    f << "ARMA_FLD_BIN" << '\n';
    f << x.n_rows << '\n';
    f << x.n_cols << '\n';
    }
  else
    {
    f << "ARMA_FL3_BIN" << '\n';
    f << x.n_rows   << '\n';
    f << x.n_cols   << '\n';
    f << x.n_slices << '\n';
    }
  
  bool save_okay = true;
  
  for(uword i=0; i<x.n_elem; ++i)
    {
    save_okay = diskio::save_arma_binary(x[i], f);
    
    if(save_okay == false)
      {
      break;
      }
    }
  
  return save_okay;
  }



template<typename T1>
inline
bool
diskio::load_arma_binary(field<T1>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::ifstream f( name.c_str(), std::fstream::binary );
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_arma_binary(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



template<typename T1>
inline
bool
diskio::load_arma_binary(field<T1>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( (is_Mat<T1>::value == false) && (is_Cube<T1>::value == false) ));
  
  bool load_okay = true;
  
  std::string f_type;
  f >> f_type;
  
  if(f_type == "ARMA_FLD_BIN")
    {
    uword f_n_rows;
    uword f_n_cols;
  
    f >> f_n_rows;
    f >> f_n_cols;
    
    x.set_size(f_n_rows, f_n_cols);
    
    f.get();      
    
    for(uword i=0; i<x.n_elem; ++i)
      {
      load_okay = diskio::load_arma_binary(x[i], f, err_msg);
      
      if(load_okay == false)
        {
        break;
        }
      }
    }
  else
  if(f_type == "ARMA_FL3_BIN")
    {
    uword f_n_rows;
    uword f_n_cols;
    uword f_n_slices;
  
    f >> f_n_rows;
    f >> f_n_cols;
    f >> f_n_slices;
    
    x.set_size(f_n_rows, f_n_cols, f_n_slices);
    
    f.get();      
    
    for(uword i=0; i<x.n_elem; ++i)
      {
      load_okay = diskio::load_arma_binary(x[i], f, err_msg);
      
      if(load_okay == false)
        {
        break;
        }
      }
    }
  else
    {
    load_okay = false;
    err_msg = "unsupported field type in ";
    }
  
  return load_okay;
  }



inline
bool
diskio::save_std_string(const field<std::string>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::ofstream f( tmp_name.c_str(), std::fstream::binary );
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = diskio::save_std_string(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



inline
bool
diskio::save_std_string(const field<std::string>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  for(uword row=0; row<x.n_rows; ++row)
  for(uword col=0; col<x.n_cols; ++col)
    {
    f << x.at(row,col);
    
    if(col < x.n_cols-1)
      {
      f << ' ';
      }
    else
      {
      f << '\n';
      }
    }
  
  return f.good();
  }



inline
bool
diskio::load_std_string(field<std::string>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::ifstream f( name.c_str() );
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_std_string(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



inline
bool
diskio::load_std_string(field<std::string>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  bool load_okay = true;
  
  //
  // work out the size
  
  uword f_n_rows = 0;
  uword f_n_cols = 0;
  
  bool f_n_cols_found = false;
  
  std::string line_string;
  std::string token;
  
  while( (f.good() == true) && (load_okay == true) )
    {
    std::getline(f, line_string);
    if(line_string.size() == 0)
      break;
    
    std::stringstream line_stream(line_string);
    
    uword line_n_cols = 0;
    while (line_stream >> token)
      line_n_cols++;
    
    if(f_n_cols_found == false)
      {
      f_n_cols = line_n_cols;
      f_n_cols_found = true;
      }
    else
      {
      if(line_n_cols != f_n_cols)
        {
        load_okay = false;
        err_msg = "inconsistent number of columns in ";
        }
      }
    
    ++f_n_rows;
    }
    
  if(load_okay == true)
    {
    f.clear();
    f.seekg(0, ios::beg);
    //f.seekg(start);
    
    x.set_size(f_n_rows, f_n_cols);
  
    for(uword row=0; row < x.n_rows; ++row)
      {
      for(uword col=0; col < x.n_cols; ++col)
        {
        f >> x.at(row,col);
        }
      }
    }
  
  if(f.good() == false)
    {
    load_okay = false; 
    }
  
  return load_okay;
  }



//! Try to load a field by automatically determining its type
template<typename T1>
inline
bool
diskio::load_auto_detect(field<T1>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::fstream f;
  f.open(name.c_str(), std::fstream::in | std::fstream::binary);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_auto_detect(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



//! Try to load a field by automatically determining its type
template<typename T1>
inline
bool
diskio::load_auto_detect(field<T1>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_Mat<T1>::value == false ));
  
  static const std::string ARMA_FLD_BIN = "ARMA_FLD_BIN";
  static const std::string ARMA_FL3_BIN = "ARMA_FL3_BIN";
  static const std::string           P6 = "P6";
  
  podarray<char> raw_header(uword(ARMA_FLD_BIN.length()) + 1);
  
  std::streampos pos = f.tellg();
  
  f.read( raw_header.memptr(), std::streamsize(ARMA_FLD_BIN.length()) );
  
  f.clear();
  f.seekg(pos);
  
  raw_header[uword(ARMA_FLD_BIN.length())] = '\0';
  
  const std::string header = raw_header.mem;
  
  if(ARMA_FLD_BIN == header.substr(0, ARMA_FLD_BIN.length()))
    {
    return load_arma_binary(x, f, err_msg);
    }
  else
  if(ARMA_FL3_BIN == header.substr(0, ARMA_FL3_BIN.length()))
    {
    return load_arma_binary(x, f, err_msg);
    }
  else
  if(P6 == header.substr(0, P6.length()))
    {
    return load_ppm_binary(x, f, err_msg);
    }
  else
    {
    err_msg = "unsupported header in ";
    return false;
    }
  }



//
// handling of PPM images by cubes


template<typename eT>
inline
bool
diskio::load_ppm_binary(Cube<eT>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::fstream f;
  f.open(name.c_str(), std::fstream::in | std::fstream::binary);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_ppm_binary(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



template<typename eT>
inline
bool
diskio::load_ppm_binary(Cube<eT>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  bool load_okay = true;
  
  std::string f_header;
  f >> f_header;
  
  if(f_header == "P6")
    {
    uword f_n_rows = 0;
    uword f_n_cols = 0;
    int f_maxval = 0;
  
    diskio::pnm_skip_comments(f);
  
    f >> f_n_cols;
    diskio::pnm_skip_comments(f);
  
    f >> f_n_rows;
    diskio::pnm_skip_comments(f);
  
    f >> f_maxval;
    f.get();
    
    if( (f_maxval > 0) || (f_maxval <= 65535) )
      {
      x.set_size(f_n_rows, f_n_cols, 3);
      
      if(f_maxval <= 255)
        {
        const uword n_elem = 3*f_n_cols*f_n_rows;
        podarray<u8> tmp(n_elem);
        
        f.read( reinterpret_cast<char*>(tmp.memptr()), std::streamsize(n_elem) );
        
        uword i = 0;
        
        //cout << "f_n_cols = " << f_n_cols << endl;
        //cout << "f_n_rows = " << f_n_rows << endl;
        
        
        for(uword row=0; row < f_n_rows; ++row)
          {
          for(uword col=0; col < f_n_cols; ++col)
            {
            x.at(row,col,0) = eT(tmp[i+0]);
            x.at(row,col,1) = eT(tmp[i+1]);
            x.at(row,col,2) = eT(tmp[i+2]);
            i+=3;
            }
          
          }
        }
      else
        {
        const uword n_elem = 3*f_n_cols*f_n_rows;
        podarray<u16> tmp(n_elem);
        
        f.read( reinterpret_cast<char *>(tmp.memptr()), std::streamsize(2*n_elem) );
        
        uword i = 0;
        
        for(uword row=0; row < f_n_rows; ++row)
          {
          for(uword col=0; col < f_n_cols; ++col)
            {
            x.at(row,col,0) = eT(tmp[i+0]);
            x.at(row,col,1) = eT(tmp[i+1]);
            x.at(row,col,2) = eT(tmp[i+2]);
            i+=3;
            }
          
          }
        
        }
      
      }
    else
      {
      load_okay = false;
      err_msg = "currently no code available to handle loading ";
      }
      
    if(f.good() == false)
      {
      load_okay = false;
      }
    
    }
  else
    {
    load_okay = false;
    err_msg = "unsupported header in ";
    }
  
  return load_okay;
  }



template<typename eT>
inline
bool
diskio::save_ppm_binary(const Cube<eT>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  
  std::ofstream f( tmp_name.c_str(), std::fstream::binary );
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = diskio::save_ppm_binary(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



template<typename eT>
inline
bool
diskio::save_ppm_binary(const Cube<eT>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (x.n_slices != 3), "diskio::save_ppm_binary(): given cube must have exactly 3 slices" );
  
  const uword n_elem = 3 * x.n_rows * x.n_cols;
  podarray<u8> tmp(n_elem);
  
  uword i = 0;
  for(uword row=0; row < x.n_rows; ++row)
    {
    for(uword col=0; col < x.n_cols; ++col)
      {
      tmp[i+0] = u8( access::tmp_real( x.at(row,col,0) ) );
      tmp[i+1] = u8( access::tmp_real( x.at(row,col,1) ) );
      tmp[i+2] = u8( access::tmp_real( x.at(row,col,2) ) );
      
      i+=3;
      }
    }
  
  f << "P6" << '\n';
  f << x.n_cols << '\n';
  f << x.n_rows << '\n';
  f << 255 << '\n';
  
  f.write( reinterpret_cast<const char*>(tmp.mem), std::streamsize(n_elem) );
  
  return f.good();
  }



//
// handling of PPM images by fields



template<typename T1>
inline
bool
diskio::load_ppm_binary(field<T1>& x, const std::string& name, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  std::fstream f;
  f.open(name.c_str(), std::fstream::in | std::fstream::binary);
  
  bool load_okay = f.is_open();
  
  if(load_okay == true)
    {
    load_okay = diskio::load_ppm_binary(x, f, err_msg);
    f.close();
    }
  
  return load_okay;
  }



template<typename T1>
inline
bool
diskio::load_ppm_binary(field<T1>& x, std::istream& f, std::string& err_msg)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_Mat<T1>::value == false ));
  typedef typename T1::elem_type eT;
  
  bool load_okay = true;
  
  std::string f_header;
  f >> f_header;
  
  if(f_header == "P6")
    {
    uword f_n_rows = 0;
    uword f_n_cols = 0;
    int f_maxval = 0;
  
    diskio::pnm_skip_comments(f);
  
    f >> f_n_cols;
    diskio::pnm_skip_comments(f);
  
    f >> f_n_rows;
    diskio::pnm_skip_comments(f);
  
    f >> f_maxval;
    f.get();
    
    if( (f_maxval > 0) || (f_maxval <= 65535) )
      {
      x.set_size(3);
      Mat<eT>& R = x(0);
      Mat<eT>& G = x(1);
      Mat<eT>& B = x(2);
      
      R.set_size(f_n_rows,f_n_cols);
      G.set_size(f_n_rows,f_n_cols);
      B.set_size(f_n_rows,f_n_cols);
      
      if(f_maxval <= 255)
        {
        const uword n_elem = 3*f_n_cols*f_n_rows;
        podarray<u8> tmp(n_elem);
        
        f.read( reinterpret_cast<char*>(tmp.memptr()), std::streamsize(n_elem) );
        
        uword i = 0;
        
        //cout << "f_n_cols = " << f_n_cols << endl;
        //cout << "f_n_rows = " << f_n_rows << endl;
        
        
        for(uword row=0; row < f_n_rows; ++row)
          {
          for(uword col=0; col < f_n_cols; ++col)
            {
            R.at(row,col) = eT(tmp[i+0]);
            G.at(row,col) = eT(tmp[i+1]);
            B.at(row,col) = eT(tmp[i+2]);
            i+=3;
            }
          
          }
        }
      else
        {
        const uword n_elem = 3*f_n_cols*f_n_rows;
        podarray<u16> tmp(n_elem);
        
        f.read( reinterpret_cast<char *>(tmp.memptr()), std::streamsize(2*n_elem) );
        
        uword i = 0;
        
        for(uword row=0; row < f_n_rows; ++row)
          {
          for(uword col=0; col < f_n_cols; ++col)
            {
            R.at(row,col) = eT(tmp[i+0]);
            G.at(row,col) = eT(tmp[i+1]);
            B.at(row,col) = eT(tmp[i+2]);
            i+=3;
            }
          
          }
        
        }
      
      }
    else
      {
      load_okay = false;
      err_msg = "currently no code available to handle loading ";
      }
    
    if(f.good() == false)
      {
      load_okay = false;
      }
    
    }
  else
    {
    load_okay = false;
    err_msg = "unsupported header in ";
    }
  
  return load_okay;
  }



template<typename T1>
inline
bool
diskio::save_ppm_binary(const field<T1>& x, const std::string& final_name)
  {
  arma_extra_debug_sigprint();
  
  const std::string tmp_name = diskio::gen_tmp_name(final_name);
  std::ofstream f( tmp_name.c_str(), std::fstream::binary );
  
  bool save_okay = f.is_open();
  
  if(save_okay == true)
    {
    save_okay = diskio::save_ppm_binary(x, f);
    
    f.flush();
    f.close();
    
    if(save_okay == true)
      {
      save_okay = diskio::safe_rename(tmp_name, final_name);
      }
    }
  
  return save_okay;
  }



template<typename T1>
inline
bool
diskio::save_ppm_binary(const field<T1>& x, std::ostream& f)
  {
  arma_extra_debug_sigprint();
  
  arma_type_check(( is_Mat<T1>::value == false ));
  
  typedef typename T1::elem_type eT;
  
  arma_debug_check( (x.n_elem != 3), "diskio::save_ppm_binary(): given field must have exactly 3 matrices of equal size" );
  
  bool same_size = true;
  for(uword i=1; i<3; ++i)
    {
    if( (x(0).n_rows != x(i).n_rows) || (x(0).n_cols != x(i).n_cols) )
      {
      same_size = false;
      break;
      }
    }
  
  arma_debug_check( (same_size != true), "diskio::save_ppm_binary(): given field must have exactly 3 matrices of equal size" );
  
  const Mat<eT>& R = x(0);
  const Mat<eT>& G = x(1);
  const Mat<eT>& B = x(2);
  
  f << "P6" << '\n';
  f << R.n_cols << '\n';
  f << R.n_rows << '\n';
  f << 255 << '\n';

  const uword n_elem = 3 * R.n_rows * R.n_cols;
  podarray<u8> tmp(n_elem);

  uword i = 0;
  for(uword row=0; row < R.n_rows; ++row)
    {
    for(uword col=0; col < R.n_cols; ++col)
      {
      tmp[i+0] = u8( access::tmp_real( R.at(row,col) ) );
      tmp[i+1] = u8( access::tmp_real( G.at(row,col) ) );
      tmp[i+2] = u8( access::tmp_real( B.at(row,col) ) );
      
      i+=3;
      }
    }
  
  f.write( reinterpret_cast<const char*>(tmp.mem), std::streamsize(n_elem) );
  
  return f.good();
  }



//! @}

