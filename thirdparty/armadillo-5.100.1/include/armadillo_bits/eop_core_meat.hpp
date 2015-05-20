// Copyright (C) 2010-2015 Conrad Sanderson
// Copyright (C) 2010-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup eop_core
//! @{


#undef arma_applier_1u
#undef arma_applier_1a
#undef arma_applier_2
#undef arma_applier_3
#undef operatorA


#if defined(ARMA_SIMPLE_LOOPS)
  #define arma_applier_1u(operatorA) \
    {\
    for(uword i=0; i<n_elem; ++i)\
      {\
      out_mem[i] operatorA eop_core<eop_type>::process(P[i], k);\
      }\
    }
#else
  #define arma_applier_1u(operatorA) \
    {\
    uword i,j;\
    \
    for(i=0, j=1; j<n_elem; i+=2, j+=2)\
      {\
      eT tmp_i = P[i];\
      eT tmp_j = P[j];\
      \
      tmp_i = eop_core<eop_type>::process(tmp_i, k);\
      tmp_j = eop_core<eop_type>::process(tmp_j, k);\
      \
      out_mem[i] operatorA tmp_i;\
      out_mem[j] operatorA tmp_j;\
      }\
    \
    if(i < n_elem)\
      {\
      out_mem[i] operatorA eop_core<eop_type>::process(P[i], k);\
      }\
    }
#endif


#if defined(ARMA_SIMPLE_LOOPS)
  #define arma_applier_1a(operatorA) \
    {\
    for(uword i=0; i<n_elem; ++i)\
      {\
      out_mem[i] operatorA eop_core<eop_type>::process(P.at_alt(i), k);\
      }\
    }
#else
  #define arma_applier_1a(operatorA) \
    {\
    uword i,j;\
    \
    for(i=0, j=1; j<n_elem; i+=2, j+=2)\
      {\
      eT tmp_i = P.at_alt(i);\
      eT tmp_j = P.at_alt(j);\
      \
      tmp_i = eop_core<eop_type>::process(tmp_i, k);\
      tmp_j = eop_core<eop_type>::process(tmp_j, k);\
      \
      out_mem[i] operatorA tmp_i;\
      out_mem[j] operatorA tmp_j;\
      }\
    \
    if(i < n_elem)\
      {\
      out_mem[i] operatorA eop_core<eop_type>::process(P.at_alt(i), k);\
      }\
    }
#endif


#define arma_applier_2(operatorA) \
  {\
  if(n_rows != 1)\
    {\
    for(uword col=0; col<n_cols; ++col)\
      {\
      uword i,j;\
      \
      for(i=0, j=1; j<n_rows; i+=2, j+=2)\
        {\
        eT tmp_i = P.at(i,col);\
        eT tmp_j = P.at(j,col);\
        \
        tmp_i = eop_core<eop_type>::process(tmp_i, k);\
        tmp_j = eop_core<eop_type>::process(tmp_j, k);\
        \
        *out_mem operatorA tmp_i;  out_mem++;\
        *out_mem operatorA tmp_j;  out_mem++;\
        }\
      \
      if(i < n_rows)\
        {\
        *out_mem operatorA eop_core<eop_type>::process(P.at(i,col), k);  out_mem++;\
        }\
      }\
    }\
  else\
    {\
    for(uword count=0; count < n_cols; ++count)\
      {\
      out_mem[count] operatorA eop_core<eop_type>::process(P.at(0,count), k);\
      }\
    }\
  }



#define arma_applier_3(operatorA) \
  {\
  for(uword slice=0; slice<n_slices; ++slice)\
    {\
    for(uword col=0; col<n_cols; ++col)\
      {\
      uword i,j;\
      \
      for(i=0, j=1; j<n_rows; i+=2, j+=2)\
        {\
        eT tmp_i = P.at(i,col,slice);\
        eT tmp_j = P.at(j,col,slice);\
        \
        tmp_i = eop_core<eop_type>::process(tmp_i, k);\
        tmp_j = eop_core<eop_type>::process(tmp_j, k);\
        \
        *out_mem operatorA tmp_i; out_mem++; \
        *out_mem operatorA tmp_j; out_mem++; \
        }\
      \
      if(i < n_rows)\
        {\
        *out_mem operatorA eop_core<eop_type>::process(P.at(i,col,slice), k); out_mem++; \
        }\
      }\
    }\
  }



//
// matrices



template<typename eop_type>
template<typename outT, typename T1>
arma_hot
inline
void
eop_core<eop_type>::apply(outT& out, const eOp<T1, eop_type>& x)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  // NOTE: we're assuming that the matrix has already been set to the correct size and there is no aliasing;
  // size setting and alias checking is done by either the Mat contructor or operator=()
  
  const eT  k       = x.aux;
        eT* out_mem = out.memptr();
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    const uword n_elem = x.get_n_elem();
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      if(x.P.is_aligned())
        {
        typename Proxy<T1>::aligned_ea_type P = x.P.get_aligned_ea();
        
        arma_applier_1a(=);
        }
      else
        {
        typename Proxy<T1>::ea_type P = x.P.get_ea();
        
        arma_applier_1u(=);
        }
      }
    else
      {
      typename Proxy<T1>::ea_type P = x.P.get_ea();
      
      arma_applier_1u(=);
      }
    }
  else
    {
    const uword n_rows = x.get_n_rows();
    const uword n_cols = x.get_n_cols();
    
    const Proxy<T1>& P = x.P;
    
    arma_applier_2(=);
    }
  }



template<typename eop_type>
template<typename T1>
arma_hot
inline
void
eop_core<eop_type>::apply_inplace_plus(Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_rows = x.get_n_rows();
  const uword n_cols = x.get_n_cols();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, n_rows, n_cols, "addition");
  
  const eT  k       = x.aux;
        eT* out_mem = out.memptr();
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    const uword n_elem = x.get_n_elem();
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      if(x.P.is_aligned())
        {
        typename Proxy<T1>::aligned_ea_type P = x.P.get_aligned_ea();
        
        arma_applier_1a(+=);
        }
      else
        {
        typename Proxy<T1>::ea_type P = x.P.get_ea();
        
        arma_applier_1u(+=);
        }
      }
    else
      {
      typename Proxy<T1>::ea_type P = x.P.get_ea();
      
      arma_applier_1u(+=);
      }
    }
  else
    {
    const Proxy<T1>& P = x.P;
    
    arma_applier_2(+=);
    }
  }



template<typename eop_type>
template<typename T1>
arma_hot
inline
void
eop_core<eop_type>::apply_inplace_minus(Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_rows = x.get_n_rows();
  const uword n_cols = x.get_n_cols();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, n_rows, n_cols, "subtraction");
  
  const eT  k       = x.aux;
        eT* out_mem = out.memptr();
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    const uword n_elem = x.get_n_elem();
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      if(x.P.is_aligned())
        {
        typename Proxy<T1>::aligned_ea_type P = x.P.get_aligned_ea();
        
        arma_applier_1a(-=);
        }
      else
        {
        typename Proxy<T1>::ea_type P = x.P.get_ea();
        
        arma_applier_1u(-=);
        }
      }
    else
      {
      typename Proxy<T1>::ea_type P = x.P.get_ea();
      
      arma_applier_1u(-=);
      }
    }
  else
    {
    const Proxy<T1>& P = x.P;
    
    arma_applier_2(-=);
    }
  }



template<typename eop_type>
template<typename T1>
arma_hot
inline
void
eop_core<eop_type>::apply_inplace_schur(Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_rows = x.get_n_rows();
  const uword n_cols = x.get_n_cols();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, n_rows, n_cols, "element-wise multiplication");
  
  const eT  k       = x.aux;
        eT* out_mem = out.memptr();
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    const uword n_elem = x.get_n_elem();
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      if(x.P.is_aligned())
        {
        typename Proxy<T1>::aligned_ea_type P = x.P.get_aligned_ea();
        
        arma_applier_1a(*=);
        }
      else
        {
        typename Proxy<T1>::ea_type P = x.P.get_ea();
        
        arma_applier_1u(*=);
        }
      }
    else
      {
      typename Proxy<T1>::ea_type P = x.P.get_ea();
      
      arma_applier_1u(*=);
      }
    }
  else
    {
    const Proxy<T1>& P = x.P;
    
    arma_applier_2(*=);
    }
  }



template<typename eop_type>
template<typename T1>
arma_hot
inline
void
eop_core<eop_type>::apply_inplace_div(Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_rows = x.get_n_rows();
  const uword n_cols = x.get_n_cols();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, n_rows, n_cols, "element-wise division");
  
  const eT  k       = x.aux;
        eT* out_mem = out.memptr();
  
  if(Proxy<T1>::prefer_at_accessor == false)
    {
    const uword n_elem = x.get_n_elem();
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      if(x.P.is_aligned())
        {
        typename Proxy<T1>::aligned_ea_type P = x.P.get_aligned_ea();
        
        arma_applier_1a(/=);
        }
      else
        {
        typename Proxy<T1>::ea_type P = x.P.get_ea();
        
        arma_applier_1u(/=);
        }
      }
    else
      {
      typename Proxy<T1>::ea_type P = x.P.get_ea();
      
      arma_applier_1u(/=);
      }
    }
  else
    {
    const Proxy<T1>& P = x.P;
    
    arma_applier_2(/=);
    }
  }



//
// cubes



template<typename eop_type>
template<typename T1>
arma_hot
inline
void
eop_core<eop_type>::apply(Cube<typename T1::elem_type>& out, const eOpCube<T1, eop_type>& x)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  // NOTE: we're assuming that the matrix has already been set to the correct size and there is no aliasing;
  // size setting and alias checking is done by either the Mat contructor or operator=()
  
  const eT  k       = x.aux;
        eT* out_mem = out.memptr();
  
  if(ProxyCube<T1>::prefer_at_accessor == false)
    {
    const uword n_elem = out.n_elem;
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      if(x.P.is_aligned())
        {
        typename ProxyCube<T1>::aligned_ea_type P = x.P.get_aligned_ea();
        
        arma_applier_1a(=);
        }
      else
        {
        typename ProxyCube<T1>::ea_type P = x.P.get_ea();
        
        arma_applier_1u(=);
        }
      }
    else
      {
      typename ProxyCube<T1>::ea_type P = x.P.get_ea();
      
      arma_applier_1u(=);
      }
    }
  else
    {
    const uword n_rows   = x.get_n_rows();
    const uword n_cols   = x.get_n_cols();
    const uword n_slices = x.get_n_slices();
    
    const ProxyCube<T1>& P = x.P;
    
    arma_applier_3(=);
    }
  }



template<typename eop_type>
template<typename T1>
arma_hot
inline
void
eop_core<eop_type>::apply_inplace_plus(Cube<typename T1::elem_type>& out, const eOpCube<T1, eop_type>& x)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_rows   = x.get_n_rows();
  const uword n_cols   = x.get_n_cols();
  const uword n_slices = x.get_n_slices();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, out.n_slices, n_rows, n_cols, n_slices, "addition");
  
  const eT  k       = x.aux;
        eT* out_mem = out.memptr();
  
  if(ProxyCube<T1>::prefer_at_accessor == false)
    {
    const uword n_elem = out.n_elem;
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      if(x.P.is_aligned())
        {
        typename ProxyCube<T1>::aligned_ea_type P = x.P.get_aligned_ea();
        
        arma_applier_1a(+=);
        }
      else
        {
        typename ProxyCube<T1>::ea_type P = x.P.get_ea();
        
        arma_applier_1u(+=);
        }
      }
    else
      {
      typename ProxyCube<T1>::ea_type P = x.P.get_ea();
      
      arma_applier_1u(+=);
      }
    }
  else
    {
    const ProxyCube<T1>& P = x.P;
    
    arma_applier_3(+=);
    }
  }



template<typename eop_type>
template<typename T1>
arma_hot
inline
void
eop_core<eop_type>::apply_inplace_minus(Cube<typename T1::elem_type>& out, const eOpCube<T1, eop_type>& x)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_rows   = x.get_n_rows();
  const uword n_cols   = x.get_n_cols();
  const uword n_slices = x.get_n_slices();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, out.n_slices, n_rows, n_cols, n_slices, "subtraction");
  
  const eT  k       = x.aux;
        eT* out_mem = out.memptr();
  
  if(ProxyCube<T1>::prefer_at_accessor == false)
    {
    const uword n_elem = out.n_elem;
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      if(x.P.is_aligned())
        {
        typename ProxyCube<T1>::aligned_ea_type P = x.P.get_aligned_ea();
        
        arma_applier_1a(-=);
        }
      else
        {
        typename ProxyCube<T1>::ea_type P = x.P.get_ea();
        
        arma_applier_1u(-=);
        }
      }
    else
      {
      typename ProxyCube<T1>::ea_type P = x.P.get_ea();
      
      arma_applier_1u(-=);
      }
    }
  else
    {
    const ProxyCube<T1>& P = x.P;
    
    arma_applier_3(-=);
    }
  }



template<typename eop_type>
template<typename T1>
arma_hot
inline
void
eop_core<eop_type>::apply_inplace_schur(Cube<typename T1::elem_type>& out, const eOpCube<T1, eop_type>& x)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_rows   = x.get_n_rows();
  const uword n_cols   = x.get_n_cols();
  const uword n_slices = x.get_n_slices();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, out.n_slices, n_rows, n_cols, n_slices, "element-wise multiplication");
  
  const eT  k       = x.aux;
        eT* out_mem = out.memptr();
  
  if(ProxyCube<T1>::prefer_at_accessor == false)
    {
    const uword n_elem = out.n_elem;
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      if(x.P.is_aligned())
        {
        typename ProxyCube<T1>::aligned_ea_type P = x.P.get_aligned_ea();
        
        arma_applier_1a(*=);
        }
      else
        {
        typename ProxyCube<T1>::ea_type P = x.P.get_ea();
        
        arma_applier_1u(*=);
        }
      }
    else
      {
      typename ProxyCube<T1>::ea_type P = x.P.get_ea();
      
      arma_applier_1u(*=);
      }
    }
  else
    {
    const ProxyCube<T1>& P = x.P;
    
    arma_applier_3(*=);
    }
  }



template<typename eop_type>
template<typename T1>
arma_hot
inline
void
eop_core<eop_type>::apply_inplace_div(Cube<typename T1::elem_type>& out, const eOpCube<T1, eop_type>& x)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword n_rows   = x.get_n_rows();
  const uword n_cols   = x.get_n_cols();
  const uword n_slices = x.get_n_slices();
  
  arma_debug_assert_same_size(out.n_rows, out.n_cols, out.n_slices, n_rows, n_cols, n_slices, "element-wise division");
  
  const eT  k       = x.aux;
        eT* out_mem = out.memptr();
  
  if(ProxyCube<T1>::prefer_at_accessor == false)
    {
    const uword n_elem = out.n_elem;
    
    if(memory::is_aligned(out_mem))
      {
      memory::mark_as_aligned(out_mem);
      
      if(x.P.is_aligned())
        {
        typename ProxyCube<T1>::aligned_ea_type P = x.P.get_aligned_ea();
        
        arma_applier_1a(/=);
        }
      else
        {
        typename ProxyCube<T1>::ea_type P = x.P.get_ea();
        
        arma_applier_1u(/=);
        }
      }
    else
      {
      typename ProxyCube<T1>::ea_type P = x.P.get_ea();
      
      arma_applier_1u(/=);
      }
    }
  else
    {
    const ProxyCube<T1>& P = x.P;
    
    arma_applier_3(/=);
    }
  }



//
// common



template<typename eop_type>
template<typename eT>
arma_hot
arma_pure
arma_inline
eT
eop_core<eop_type>::process(const eT, const eT)
  {
  arma_stop("eop_core::process(): unhandled eop_type");
  return eT(0);
  }



template<> template<typename eT> arma_hot arma_const arma_inline eT
eop_core<eop_scalar_plus      >::process(const eT val, const eT k) { return val + k;                  }

template<> template<typename eT> arma_hot arma_const arma_inline eT
eop_core<eop_scalar_minus_pre >::process(const eT val, const eT k) { return k - val;                  }

template<> template<typename eT> arma_hot arma_const arma_inline eT
eop_core<eop_scalar_minus_post>::process(const eT val, const eT k) { return val - k;                  }

template<> template<typename eT> arma_hot arma_const arma_inline eT
eop_core<eop_scalar_times     >::process(const eT val, const eT k) { return val * k;                  }

template<> template<typename eT> arma_hot arma_const arma_inline eT
eop_core<eop_scalar_div_pre   >::process(const eT val, const eT k) { return k / val;                  }

template<> template<typename eT> arma_hot arma_const arma_inline eT
eop_core<eop_scalar_div_post  >::process(const eT val, const eT k) { return val / k;                  }

template<> template<typename eT> arma_hot arma_const arma_inline eT
eop_core<eop_square           >::process(const eT val, const eT  ) { return val*val;                  }

template<> template<typename eT> arma_hot arma_const arma_inline eT
eop_core<eop_neg              >::process(const eT val, const eT  ) { return eop_aux::neg(val);        }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_sqrt             >::process(const eT val, const eT  ) { return eop_aux::sqrt(val);       }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_log              >::process(const eT val, const eT  ) { return eop_aux::log(val);        }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_log2             >::process(const eT val, const eT  ) { return eop_aux::log2(val);       }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_log10            >::process(const eT val, const eT  ) { return eop_aux::log10(val);      }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_trunc_log        >::process(const eT val, const eT  ) { return    arma::trunc_log(val);  }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_exp              >::process(const eT val, const eT  ) { return eop_aux::exp(val);        }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_exp2             >::process(const eT val, const eT  ) { return eop_aux::exp2(val);       }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_exp10            >::process(const eT val, const eT  ) { return eop_aux::exp10(val);      }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_trunc_exp        >::process(const eT val, const eT  ) { return    arma::trunc_exp(val);  }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_cos              >::process(const eT val, const eT  ) { return eop_aux::cos(val);        }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_sin              >::process(const eT val, const eT  ) { return eop_aux::sin(val);        }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_tan              >::process(const eT val, const eT  ) { return eop_aux::tan(val);        }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_acos             >::process(const eT val, const eT  ) { return eop_aux::acos(val);       }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_asin             >::process(const eT val, const eT  ) { return eop_aux::asin(val);       }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_atan             >::process(const eT val, const eT  ) { return eop_aux::atan(val);       }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_cosh             >::process(const eT val, const eT  ) { return eop_aux::cosh(val);       }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_sinh             >::process(const eT val, const eT  ) { return eop_aux::sinh(val);       }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_tanh             >::process(const eT val, const eT  ) { return eop_aux::tanh(val);       }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_acosh            >::process(const eT val, const eT  ) { return eop_aux::acosh(val);      }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_asinh            >::process(const eT val, const eT  ) { return eop_aux::asinh(val);      }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_atanh            >::process(const eT val, const eT  ) { return eop_aux::atanh(val);      }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_eps              >::process(const eT val, const eT  ) { return eop_aux::direct_eps(val); }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_abs              >::process(const eT val, const eT  ) { return eop_aux::arma_abs(val);   }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_conj             >::process(const eT val, const eT  ) { return eop_aux::conj(val);       }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_pow              >::process(const eT val, const eT k) { return eop_aux::pow(val, k);     }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_floor            >::process(const eT val, const eT  ) { return eop_aux::floor(val);      }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_ceil             >::process(const eT val, const eT  ) { return eop_aux::ceil(val);       }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_round            >::process(const eT val, const eT  ) { return eop_aux::round(val);      }

template<> template<typename eT> arma_hot arma_pure arma_inline eT
eop_core<eop_sign             >::process(const eT val, const eT  ) { return eop_aux::sign(val);       }


#undef arma_applier_1u
#undef arma_applier_1a
#undef arma_applier_2
#undef arma_applier_3



//! @}
