// Copyright (C) 2008-2015 Conrad Sanderson
// Copyright (C) 2008-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup op_sort_index
//! @{



class op_sort_index
  {
  public:
  
  template<typename T1>
  static inline bool apply_noalias(Mat<uword>& out, const Proxy<T1>& P, const uword sort_type);
  
  template<typename T1>
  static inline void apply(Mat<uword>& out, const mtOp<uword,T1,op_sort_index>& in);
  };



class op_stable_sort_index
  {
  public:
  
  template<typename T1>
  static inline bool apply_noalias(Mat<uword>& out, const Proxy<T1>& P, const uword sort_type);
  
  template<typename T1>
  static inline void apply(Mat<uword>& out, const mtOp<uword,T1,op_stable_sort_index>& in);
  };



template<typename T1, typename T2>
struct arma_sort_index_packet
  {
  T1 val;
  T2 index;
  };



class arma_sort_index_helper_ascend
  {
  public:
  
  template<typename T1, typename T2>
  arma_inline
  bool
  operator() (const arma_sort_index_packet<T1,T2>& A, const arma_sort_index_packet<T1,T2>& B) const
    {
    return (A.val < B.val);
    }
  };



class arma_sort_index_helper_descend
  {
  public:
  
  template<typename T1, typename T2>
  arma_inline
  bool
  operator() (const arma_sort_index_packet<T1,T2>& A, const arma_sort_index_packet<T1,T2>& B) const
    {
    return (A.val > B.val);
    }
  };



//! @}
