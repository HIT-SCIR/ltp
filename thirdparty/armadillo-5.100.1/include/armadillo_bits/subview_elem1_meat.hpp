// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup subview_elem1
//! @{


template<typename eT, typename T1>
inline
subview_elem1<eT,T1>::~subview_elem1()
  {
  arma_extra_debug_sigprint();
  }


template<typename eT, typename T1>
arma_inline
subview_elem1<eT,T1>::subview_elem1(const Mat<eT>& in_m, const Base<uword,T1>& in_a)
  : m(in_m)
  , a(in_a)
  {
  arma_extra_debug_sigprint();
  }



template<typename eT, typename T1>
arma_inline
subview_elem1<eT,T1>::subview_elem1(const Cube<eT>& in_q, const Base<uword,T1>& in_a)
  : fake_m( const_cast< eT* >(in_q.memptr()), in_q.n_elem, 1, false )
  ,      m( fake_m )
  ,      a( in_a   )
  {
  arma_extra_debug_sigprint();
  }



template<typename eT, typename T1>
template<typename op_type>
inline
void
subview_elem1<eT,T1>::inplace_op(const eT val)
  {
  arma_extra_debug_sigprint();
  
  Mat<eT>& m_local = const_cast< Mat<eT>& >(m);
  
        eT*   m_mem    = m_local.memptr();
  const uword m_n_elem = m_local.n_elem;
  
  const unwrap_check_mixed<T1> tmp(a.get_ref(), m_local);
  const umat& aa = tmp.M;
  
  arma_debug_check
    (
    ( (aa.is_vec() == false) && (aa.is_empty() == false) ),
    "Mat::elem(): given object is not a vector"
    );
  
  const uword* aa_mem    = aa.memptr();
  const uword  aa_n_elem = aa.n_elem;
  
  uword iq,jq;
  for(iq=0, jq=1; jq < aa_n_elem; iq+=2, jq+=2)
    {
    const uword ii = aa_mem[iq];
    const uword jj = aa_mem[jq];
    
    arma_debug_check( ( (ii >= m_n_elem) || (jj >= m_n_elem) ), "Mat::elem(): index out of bounds" );
    
         if(is_same_type<op_type, op_subview_elem_equ          >::yes) { m_mem[ii] =  val; m_mem[jj] =  val; }
    else if(is_same_type<op_type, op_subview_elem_inplace_plus >::yes) { m_mem[ii] += val; m_mem[jj] += val; }
    else if(is_same_type<op_type, op_subview_elem_inplace_minus>::yes) { m_mem[ii] -= val; m_mem[jj] -= val; }
    else if(is_same_type<op_type, op_subview_elem_inplace_schur>::yes) { m_mem[ii] *= val; m_mem[jj] *= val; }
    else if(is_same_type<op_type, op_subview_elem_inplace_div  >::yes) { m_mem[ii] /= val; m_mem[jj] /= val; }
    }
  
  if(iq < aa_n_elem)
    {
    const uword ii = aa_mem[iq];
    
    arma_debug_check( (ii >= m_n_elem) , "Mat::elem(): index out of bounds" ); 
    
         if(is_same_type<op_type, op_subview_elem_equ          >::yes) { m_mem[ii] =  val; }
    else if(is_same_type<op_type, op_subview_elem_inplace_plus >::yes) { m_mem[ii] += val; }
    else if(is_same_type<op_type, op_subview_elem_inplace_minus>::yes) { m_mem[ii] -= val; }
    else if(is_same_type<op_type, op_subview_elem_inplace_schur>::yes) { m_mem[ii] *= val; }
    else if(is_same_type<op_type, op_subview_elem_inplace_div  >::yes) { m_mem[ii] /= val; }
    }
  }



template<typename eT, typename T1>
template<typename op_type, typename T2>
inline
void
subview_elem1<eT,T1>::inplace_op(const subview_elem1<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  subview_elem1<eT,T1>& s = *this;
  
  if(&(s.m) == &(x.m))
    {
    arma_extra_debug_print("subview_elem1::inplace_op(): aliasing detected");
    
    const Mat<eT> tmp(x);
    
         if(is_same_type<op_type, op_subview_elem_equ          >::yes) { s.operator= (tmp); }
    else if(is_same_type<op_type, op_subview_elem_inplace_plus >::yes) { s.operator+=(tmp); }
    else if(is_same_type<op_type, op_subview_elem_inplace_minus>::yes) { s.operator-=(tmp); }
    else if(is_same_type<op_type, op_subview_elem_inplace_schur>::yes) { s.operator%=(tmp); }
    else if(is_same_type<op_type, op_subview_elem_inplace_div  >::yes) { s.operator/=(tmp); }
    }
  else
    {
          Mat<eT>& s_m_local = const_cast< Mat<eT>& >(s.m);
    const Mat<eT>& x_m_local = x.m;
    
    const unwrap_check_mixed<T1> s_tmp(s.a.get_ref(), s_m_local);
    const unwrap_check_mixed<T2> x_tmp(x.a.get_ref(), s_m_local);
    
    const umat& s_aa = s_tmp.M;
    const umat& x_aa = x_tmp.M;
    
    arma_debug_check
      (
      ( ((s_aa.is_vec() == false) && (s_aa.is_empty() == false)) || ((x_aa.is_vec() == false) && (x_aa.is_empty() == false)) ),
      "Mat::elem(): given object is not a vector"
      );
    
    const uword* s_aa_mem = s_aa.memptr();
    const uword* x_aa_mem = x_aa.memptr();
    
    const uword s_aa_n_elem = s_aa.n_elem;
    
    arma_debug_check( (s_aa_n_elem != x_aa.n_elem), "Mat::elem(): size mismatch" );
    
    
          eT*   s_m_mem    = s_m_local.memptr();
    const uword s_m_n_elem = s_m_local.n_elem;
    
    const eT*   x_m_mem    = x_m_local.memptr();
    const uword x_m_n_elem = x_m_local.n_elem;
    
    uword iq,jq;
    for(iq=0, jq=1; jq < s_aa_n_elem; iq+=2, jq+=2)
      {
      const uword s_ii = s_aa_mem[iq];
      const uword s_jj = s_aa_mem[jq];
      
      const uword x_ii = x_aa_mem[iq];
      const uword x_jj = x_aa_mem[jq];
      
      arma_debug_check
        (
        (s_ii >= s_m_n_elem) || (s_jj >= s_m_n_elem) || (x_ii >= x_m_n_elem) || (x_jj >= x_m_n_elem),
        "Mat::elem(): index out of bounds"
        );
      
           if(is_same_type<op_type, op_subview_elem_equ          >::yes) { s_m_mem[s_ii]  = x_m_mem[x_ii]; s_m_mem[s_jj]  = x_m_mem[x_jj]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_plus >::yes) { s_m_mem[s_ii] += x_m_mem[x_ii]; s_m_mem[s_jj] += x_m_mem[x_jj]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_minus>::yes) { s_m_mem[s_ii] -= x_m_mem[x_ii]; s_m_mem[s_jj] -= x_m_mem[x_jj]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_schur>::yes) { s_m_mem[s_ii] *= x_m_mem[x_ii]; s_m_mem[s_jj] *= x_m_mem[x_jj]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_div  >::yes) { s_m_mem[s_ii] /= x_m_mem[x_ii]; s_m_mem[s_jj] /= x_m_mem[x_jj]; }
      }
    
    if(iq < s_aa_n_elem)
      {
      const uword s_ii = s_aa_mem[iq];
      const uword x_ii = x_aa_mem[iq];
      
      arma_debug_check
        (
        ( (s_ii >= s_m_n_elem) || (x_ii >= x_m_n_elem) ),
        "Mat::elem(): index out of bounds"
        );
      
           if(is_same_type<op_type, op_subview_elem_equ          >::yes) { s_m_mem[s_ii]  = x_m_mem[x_ii]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_plus >::yes) { s_m_mem[s_ii] += x_m_mem[x_ii]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_minus>::yes) { s_m_mem[s_ii] -= x_m_mem[x_ii]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_schur>::yes) { s_m_mem[s_ii] *= x_m_mem[x_ii]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_div  >::yes) { s_m_mem[s_ii] /= x_m_mem[x_ii]; }
      }
    }
  }



template<typename eT, typename T1>
template<typename op_type, typename T2>
inline
void
subview_elem1<eT,T1>::inplace_op(const Base<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  Mat<eT>& m_local = const_cast< Mat<eT>& >(m);
  
        eT*   m_mem    = m_local.memptr();
  const uword m_n_elem = m_local.n_elem;
  
  const unwrap_check_mixed<T1> aa_tmp(a.get_ref(), m_local);
  const umat& aa = aa_tmp.M;
  
  arma_debug_check
    (
    ( (aa.is_vec() == false) && (aa.is_empty() == false) ),
    "Mat::elem(): given object is not a vector"
    );
  
  const uword* aa_mem    = aa.memptr();
  const uword  aa_n_elem = aa.n_elem;
  
  const Proxy<T2> P(x.get_ref());
  
  arma_debug_check( (aa_n_elem != P.get_n_elem()), "Mat::elem(): size mismatch" );
  
  const bool is_alias = P.is_alias(m);
  
  if( (is_alias == false) && (Proxy<T2>::prefer_at_accessor == false) )
    {
    typename Proxy<T2>::ea_type X = P.get_ea();
    
    uword iq,jq;
    for(iq=0, jq=1; jq < aa_n_elem; iq+=2, jq+=2)
      {
      const uword ii = aa_mem[iq];
      const uword jj = aa_mem[jq];
      
      arma_debug_check( ( (ii >= m_n_elem) || (jj >= m_n_elem) ), "Mat::elem(): index out of bounds" );
      
           if(is_same_type<op_type, op_subview_elem_equ          >::yes) { m_mem[ii] =  X[iq]; m_mem[jj]  = X[jq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_plus >::yes) { m_mem[ii] += X[iq]; m_mem[jj] += X[jq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_minus>::yes) { m_mem[ii] -= X[iq]; m_mem[jj] -= X[jq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_schur>::yes) { m_mem[ii] *= X[iq]; m_mem[jj] *= X[jq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_div  >::yes) { m_mem[ii] /= X[iq]; m_mem[jj] /= X[jq]; }
      }
    
    if(iq < aa_n_elem)
      {
      const uword ii = aa_mem[iq];
      
      arma_debug_check( (ii >= m_n_elem) , "Mat::elem(): index out of bounds" );
      
           if(is_same_type<op_type, op_subview_elem_equ          >::yes) { m_mem[ii] =  X[iq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_plus >::yes) { m_mem[ii] += X[iq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_minus>::yes) { m_mem[ii] -= X[iq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_schur>::yes) { m_mem[ii] *= X[iq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_div  >::yes) { m_mem[ii] /= X[iq]; }
      }
    }
  else
    {
    arma_extra_debug_print("subview_elem1::inplace_op(): aliasing or prefer_at_accessor detected");
    
    const unwrap_check<typename Proxy<T2>::stored_type> tmp(P.Q, is_alias);
    const Mat<eT>& M = tmp.M;
    
    const eT* X = M.memptr();
    
    uword iq,jq;
    for(iq=0, jq=1; jq < aa_n_elem; iq+=2, jq+=2)
      {
      const uword ii = aa_mem[iq];
      const uword jj = aa_mem[jq];
      
      arma_debug_check( ( (ii >= m_n_elem) || (jj >= m_n_elem) ), "Mat::elem(): index out of bounds" );
      
           if(is_same_type<op_type, op_subview_elem_equ          >::yes) { m_mem[ii] =  X[iq]; m_mem[jj]  = X[jq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_plus >::yes) { m_mem[ii] += X[iq]; m_mem[jj] += X[jq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_minus>::yes) { m_mem[ii] -= X[iq]; m_mem[jj] -= X[jq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_schur>::yes) { m_mem[ii] *= X[iq]; m_mem[jj] *= X[jq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_div  >::yes) { m_mem[ii] /= X[iq]; m_mem[jj] /= X[jq]; }
      }
    
    if(iq < aa_n_elem)
      {
      const uword ii = aa_mem[iq];
      
      arma_debug_check( (ii >= m_n_elem) , "Mat::elem(): index out of bounds" );
      
           if(is_same_type<op_type, op_subview_elem_equ          >::yes) { m_mem[ii] =  X[iq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_plus >::yes) { m_mem[ii] += X[iq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_minus>::yes) { m_mem[ii] -= X[iq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_schur>::yes) { m_mem[ii] *= X[iq]; }
      else if(is_same_type<op_type, op_subview_elem_inplace_div  >::yes) { m_mem[ii] /= X[iq]; }
      }
    }
  }



//
//



template<typename eT, typename T1>
arma_inline
const Op<subview_elem1<eT,T1>,op_htrans>
subview_elem1<eT,T1>::t() const
  {
  return Op<subview_elem1<eT,T1>,op_htrans>(*this);
  }



template<typename eT, typename T1>
arma_inline
const Op<subview_elem1<eT,T1>,op_htrans>
subview_elem1<eT,T1>::ht() const
  {
  return Op<subview_elem1<eT,T1>,op_htrans>(*this);
  }



template<typename eT, typename T1>
arma_inline
const Op<subview_elem1<eT,T1>,op_strans>
subview_elem1<eT,T1>::st() const
  {
  return Op<subview_elem1<eT,T1>,op_strans>(*this);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::fill(const eT val)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_equ>(val);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::zeros()
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_equ>(eT(0));
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::ones()
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_equ>(eT(1));
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::randu()
  {
  arma_extra_debug_sigprint();
  
  Mat<eT>& m_local = const_cast< Mat<eT>& >(m);
  
        eT*   m_mem    = m_local.memptr();
  const uword m_n_elem = m_local.n_elem;
  
  const unwrap_check_mixed<T1> tmp(a.get_ref(), m_local);
  const umat& aa = tmp.M;
  
  arma_debug_check
    (
    ( (aa.is_vec() == false) && (aa.is_empty() == false) ),
    "Mat::elem(): given object is not a vector"
    );
  
  const uword* aa_mem    = aa.memptr();
  const uword  aa_n_elem = aa.n_elem;
  
  uword iq,jq;
  for(iq=0, jq=1; jq < aa_n_elem; iq+=2, jq+=2)
    {
    const uword ii = aa_mem[iq];
    const uword jj = aa_mem[jq];
    
    arma_debug_check( ( (ii >= m_n_elem) || (jj >= m_n_elem) ), "Mat::elem(): index out of bounds" );
    
    const eT val1 = eT(arma_rng::randu<eT>());
    const eT val2 = eT(arma_rng::randu<eT>());
    
    m_mem[ii] = val1;
    m_mem[jj] = val2;
    }
  
  if(iq < aa_n_elem)
    {
    const uword ii = aa_mem[iq];
    
    arma_debug_check( (ii >= m_n_elem) , "Mat::elem(): index out of bounds" ); 
    
    m_mem[ii] = eT(arma_rng::randu<eT>());
    }
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::randn()
  {
  arma_extra_debug_sigprint();
  
  Mat<eT>& m_local = const_cast< Mat<eT>& >(m);
  
        eT*   m_mem    = m_local.memptr();
  const uword m_n_elem = m_local.n_elem;
  
  const unwrap_check_mixed<T1> tmp(a.get_ref(), m_local);
  const umat& aa = tmp.M;
  
  arma_debug_check
    (
    ( (aa.is_vec() == false) && (aa.is_empty() == false) ),
    "Mat::elem(): given object is not a vector"
    );
  
  const uword* aa_mem    = aa.memptr();
  const uword  aa_n_elem = aa.n_elem;
  
  uword iq,jq;
  for(iq=0, jq=1; jq < aa_n_elem; iq+=2, jq+=2)
    {
    const uword ii = aa_mem[iq];
    const uword jj = aa_mem[jq];
    
    arma_debug_check( ( (ii >= m_n_elem) || (jj >= m_n_elem) ), "Mat::elem(): index out of bounds" );
    
    arma_rng::randn<eT>::dual_val( m_mem[ii], m_mem[jj] );
    }
  
  if(iq < aa_n_elem)
    {
    const uword ii = aa_mem[iq];
    
    arma_debug_check( (ii >= m_n_elem) , "Mat::elem(): index out of bounds" ); 
    
    m_mem[ii] = eT(arma_rng::randn<eT>());
    }
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator+= (const eT val)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_plus>(val);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator-= (const eT val)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_minus>(val);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator*= (const eT val)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_schur>(val);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator/= (const eT val)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_div>(val);
  }



//
//



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator_equ(const subview_elem1<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_equ>(x);
  }




template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator= (const subview_elem1<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  (*this).operator_equ(x);
  }



//! work around compiler bugs
template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::operator= (const subview_elem1<eT,T1>& x)
  {
  arma_extra_debug_sigprint();
  
  (*this).operator_equ(x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator+= (const subview_elem1<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_plus>(x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator-= (const subview_elem1<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_minus>(x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator%= (const subview_elem1<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_schur>(x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator/= (const subview_elem1<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_div>(x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator= (const Base<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_equ>(x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator+= (const Base<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_plus>(x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator-= (const Base<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_minus>(x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator%= (const Base<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_schur>(x);
  }



template<typename eT, typename T1>
template<typename T2>
inline
void
subview_elem1<eT,T1>::operator/= (const Base<eT,T2>& x)
  {
  arma_extra_debug_sigprint();
  
  inplace_op<op_subview_elem_inplace_div>(x);
  }



//
//



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::extract(Mat<eT>& actual_out, const subview_elem1<eT,T1>& in)
  {
  arma_extra_debug_sigprint();
  
  const unwrap_check_mixed<T1> tmp1(in.a.get_ref(), actual_out);
  const umat& aa = tmp1.M;
  
  arma_debug_check
    (
    ( (aa.is_vec() == false) && (aa.is_empty() == false) ),
    "Mat::elem(): given object is not a vector"
    );
  
  const uword* aa_mem    = aa.memptr();
  const uword  aa_n_elem = aa.n_elem;
  
  const Mat<eT>& m_local = in.m;
  
  const eT*   m_mem    = m_local.memptr();
  const uword m_n_elem = m_local.n_elem;
  
  const bool alias = (&actual_out == &m_local);
  
  arma_extra_debug_warn(alias, "subview_elem1::extract(): aliasing detected");
  
  Mat<eT>* tmp_out = alias ? new Mat<eT>() : 0;
  Mat<eT>& out     = alias ? *tmp_out      : actual_out;
  
  out.set_size(aa_n_elem, 1);
  
  eT* out_mem = out.memptr();
  
  uword i,j;
  for(i=0, j=1; j<aa_n_elem; i+=2, j+=2)
    {
    const uword ii = aa_mem[i];
    const uword jj = aa_mem[j];
    
    arma_debug_check( ( (ii >= m_n_elem) || (jj >= m_n_elem) ), "Mat::elem(): index out of bounds" );
    
    out_mem[i] = m_mem[ii];
    out_mem[j] = m_mem[jj];
    }
  
  if(i < aa_n_elem)
    {
    const uword ii = aa_mem[i];
    
    arma_debug_check( (ii >= m_n_elem) , "Mat::elem(): index out of bounds" );
    
    out_mem[i] = m_mem[ii];
    }
  
  if(alias == true)
    {
    actual_out.steal_mem(out);
    delete tmp_out;
    }
  }



template<typename eT, typename T1>
template<typename op_type>
inline
void
subview_elem1<eT,T1>::mat_inplace_op(Mat<eT>& out, const subview_elem1& in)
  {
  arma_extra_debug_sigprint();
  
  const unwrap<T1> tmp1(in.a.get_ref());
  const umat& aa = tmp1.M;
  
  arma_debug_check
    (
    ( (aa.is_vec() == false) && (aa.is_empty() == false) ),
    "Mat::elem(): given object is not a vector"
    );
  
  const uword* aa_mem    = aa.memptr();
  const uword  aa_n_elem = aa.n_elem;
  
  const unwrap_check< Mat<eT> > tmp2(in.m, out);
  const Mat<eT>& m_local      = tmp2.M;
  
  const eT*   m_mem    = m_local.memptr();
  const uword m_n_elem = m_local.n_elem;
  
  arma_debug_check( (out.n_elem != aa_n_elem), "Mat::elem(): size mismatch" );
  
  eT* out_mem = out.memptr();
  
  uword i,j;
  for(i=0, j=1; j<aa_n_elem; i+=2, j+=2)
    {
    const uword ii = aa_mem[i];
    const uword jj = aa_mem[j];
    
    arma_debug_check( ( (ii >= m_n_elem) || (jj >= m_n_elem) ), "Mat::elem(): index out of bounds" );
    
         if(is_same_type<op_type, op_subview_elem_inplace_plus >::yes) { out_mem[i] += m_mem[ii]; out_mem[j] += m_mem[jj]; }
    else if(is_same_type<op_type, op_subview_elem_inplace_minus>::yes) { out_mem[i] -= m_mem[ii]; out_mem[j] -= m_mem[jj]; }
    else if(is_same_type<op_type, op_subview_elem_inplace_schur>::yes) { out_mem[i] *= m_mem[ii]; out_mem[j] *= m_mem[jj]; }
    else if(is_same_type<op_type, op_subview_elem_inplace_div  >::yes) { out_mem[i] /= m_mem[ii]; out_mem[j] /= m_mem[jj]; }
    }
  
  if(i < aa_n_elem)
    {
    const uword ii = aa_mem[i];
    
    arma_debug_check( (ii >= m_n_elem) , "Mat::elem(): index out of bounds" );
    
         if(is_same_type<op_type, op_subview_elem_inplace_plus >::yes) { out_mem[i] += m_mem[ii]; }
    else if(is_same_type<op_type, op_subview_elem_inplace_minus>::yes) { out_mem[i] -= m_mem[ii]; }
    else if(is_same_type<op_type, op_subview_elem_inplace_schur>::yes) { out_mem[i] *= m_mem[ii]; }
    else if(is_same_type<op_type, op_subview_elem_inplace_div  >::yes) { out_mem[i] /= m_mem[ii]; }
    }
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::plus_inplace(Mat<eT>& out, const subview_elem1& in)
  {
  arma_extra_debug_sigprint();
  
  mat_inplace_op<op_subview_elem_inplace_plus>(out, in);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::minus_inplace(Mat<eT>& out, const subview_elem1& in)
  {
  arma_extra_debug_sigprint();
  
  mat_inplace_op<op_subview_elem_inplace_minus>(out, in);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::schur_inplace(Mat<eT>& out, const subview_elem1& in)
  {
  arma_extra_debug_sigprint();
  
  mat_inplace_op<op_subview_elem_inplace_schur>(out, in);
  }



template<typename eT, typename T1>
inline
void
subview_elem1<eT,T1>::div_inplace(Mat<eT>& out, const subview_elem1& in)
  {
  arma_extra_debug_sigprint();
  
  mat_inplace_op<op_subview_elem_inplace_div>(out, in);
  }



//! @}
