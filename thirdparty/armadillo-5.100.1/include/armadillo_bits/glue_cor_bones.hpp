// Copyright (C) 2009-2010 Conrad Sanderson
// Copyright (C) 2009-2010 NICTA (www.nicta.com.au)
// Copyright (C) 2009-2010 Dimitrios Bouzas
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup glue_cor
//! @{



class glue_cor
  {
  public:

  template<typename eT> inline static void direct_cor(Mat<eT>&                out, const Mat<eT>&                A, const Mat<eT>&                B, const uword norm_type);
  template<typename T>  inline static void direct_cor(Mat< std::complex<T> >& out, const Mat< std::complex<T> >& A, const Mat< std::complex<T> >& B, const uword norm_type);
  
  template<typename T1, typename T2> inline static void apply(Mat<typename T1::elem_type>& out, const Glue<T1,T2,glue_cor>& X);
  };



//! @}

