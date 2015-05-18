// Copyright (C) 2008-2014 Conrad Sanderson
// Copyright (C) 2008-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup arma_ostream
//! @{



class arma_ostream_state
  {
  private:

  const ios::fmtflags   orig_flags;
  const std::streamsize orig_precision;
  const std::streamsize orig_width;
  const char            orig_fill;


  public:

  inline arma_ostream_state(const std::ostream& o);
  
  inline void restore(std::ostream& o) const;
  };



class arma_ostream
  {
  public:
  
  template<typename eT> inline static std::streamsize modify_stream(std::ostream& o, const eT*              data, const uword n_elem);
  template<typename  T> inline static std::streamsize modify_stream(std::ostream& o, const std::complex<T>* data, const uword n_elem);
  template<typename eT> inline static std::streamsize modify_stream(std::ostream& o, typename SpMat<eT>::const_iterator begin, const uword n_elem, const typename arma_not_cx<eT>::result* junk = 0);
  template<typename  T> inline static std::streamsize modify_stream(std::ostream& o, typename SpMat<T>::const_iterator begin, const uword n_elem, const typename arma_cx_only<T>::result* junk = 0);
  
  template<typename eT> inline static void print_elem_zero(std::ostream& o, const bool modify);
  
  template<typename eT> arma_inline static void print_elem(std::ostream& o, const eT&              x, const bool modify);
  template<typename  T>      inline static void print_elem(std::ostream& o, const std::complex<T>& x, const bool modify);

  template<typename eT> inline static void print(std::ostream& o, const  Mat<eT>& m, const bool modify);
  template<typename eT> inline static void print(std::ostream& o, const Cube<eT>& m, const bool modify);
  
  template<typename oT> inline static void print(std::ostream& o, const field<oT>&         m);
  template<typename oT> inline static void print(std::ostream& o, const subview_field<oT>& m);


  template<typename eT> inline static void print_dense(std::ostream& o, const SpMat<eT>& m, const bool modify);
  template<typename eT> inline static void       print(std::ostream& o, const SpMat<eT>& m, const bool modify);
  
  inline static void print(std::ostream& o, const SizeMat&  S);
  inline static void print(std::ostream& o, const SizeCube& S);
  };



//! @}
