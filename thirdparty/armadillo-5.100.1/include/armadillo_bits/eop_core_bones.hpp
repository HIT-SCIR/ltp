// Copyright (C) 2010-2015 Conrad Sanderson
// Copyright (C) 2010-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup eop_core
//! @{



template<typename eop_type>
class eop_core
  {
  public:
  
  // matrices
  
  template<typename outT, typename T1> arma_hot inline static void apply(outT& out, const eOp<T1, eop_type>& x);
  
  template<typename T1> arma_hot inline static void apply_inplace_plus (Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x);
  template<typename T1> arma_hot inline static void apply_inplace_minus(Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x);
  template<typename T1> arma_hot inline static void apply_inplace_schur(Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x);
  template<typename T1> arma_hot inline static void apply_inplace_div  (Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x);
  
  
  // cubes
  
  template<typename T1> arma_hot inline static void apply(Cube<typename T1::elem_type>& out, const eOpCube<T1, eop_type>& x);
  
  template<typename T1> arma_hot inline static void apply_inplace_plus (Cube<typename T1::elem_type>& out, const eOpCube<T1, eop_type>& x);
  template<typename T1> arma_hot inline static void apply_inplace_minus(Cube<typename T1::elem_type>& out, const eOpCube<T1, eop_type>& x);
  template<typename T1> arma_hot inline static void apply_inplace_schur(Cube<typename T1::elem_type>& out, const eOpCube<T1, eop_type>& x);
  template<typename T1> arma_hot inline static void apply_inplace_div  (Cube<typename T1::elem_type>& out, const eOpCube<T1, eop_type>& x);
  
  
  // common
  
  template<typename eT> arma_hot arma_pure arma_inline static eT process(const eT val, const eT k);
  };



class eop_neg               : public eop_core<eop_neg>               {};
class eop_scalar_plus       : public eop_core<eop_scalar_plus>       {};
class eop_scalar_minus_pre  : public eop_core<eop_scalar_minus_pre>  {};
class eop_scalar_minus_post : public eop_core<eop_scalar_minus_post> {};
class eop_scalar_times      : public eop_core<eop_scalar_times>      {};
class eop_scalar_div_pre    : public eop_core<eop_scalar_div_pre>    {};
class eop_scalar_div_post   : public eop_core<eop_scalar_div_post>   {};
class eop_square            : public eop_core<eop_square>            {};
class eop_sqrt              : public eop_core<eop_sqrt>              {};
class eop_log               : public eop_core<eop_log>               {};
class eop_log2              : public eop_core<eop_log2>              {};
class eop_log10             : public eop_core<eop_log10>             {};
class eop_trunc_log         : public eop_core<eop_trunc_log>         {};
class eop_exp               : public eop_core<eop_exp>               {};
class eop_exp2              : public eop_core<eop_exp2>              {};
class eop_exp10             : public eop_core<eop_exp10>             {};
class eop_trunc_exp         : public eop_core<eop_trunc_exp>         {};
class eop_cos               : public eop_core<eop_cos>               {};
class eop_sin               : public eop_core<eop_sin>               {};
class eop_tan               : public eop_core<eop_tan>               {};
class eop_acos              : public eop_core<eop_acos>              {};
class eop_asin              : public eop_core<eop_asin>              {};
class eop_atan              : public eop_core<eop_atan>              {};
class eop_cosh              : public eop_core<eop_cosh>              {};
class eop_sinh              : public eop_core<eop_sinh>              {};
class eop_tanh              : public eop_core<eop_tanh>              {};
class eop_acosh             : public eop_core<eop_acosh>             {};
class eop_asinh             : public eop_core<eop_asinh>             {};
class eop_atanh             : public eop_core<eop_atanh>             {};
class eop_eps               : public eop_core<eop_eps>               {};
class eop_abs               : public eop_core<eop_abs>               {};
class eop_conj              : public eop_core<eop_conj>              {};
class eop_pow               : public eop_core<eop_pow>               {};
class eop_floor             : public eop_core<eop_floor>             {};
class eop_ceil              : public eop_core<eop_ceil>              {};
class eop_round             : public eop_core<eop_round>             {};
class eop_sign              : public eop_core<eop_sign>              {};



//! @}
