// Copyright (C) 2009-2011 Conrad Sanderson
// Copyright (C) 2009-2011 NICTA (www.nicta.com.au)
// Copyright (C) 2009-2010 Dimitrios Bouzas
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup op_cor
//! @{



template<typename eT>
inline
void
op_cor::direct_cor(Mat<eT>& out, const Mat<eT>& A, const uword norm_type)
  {
  arma_extra_debug_sigprint();
  
  if(A.is_empty())
    {
    out.reset();
    return;
    }
  
  if(A.is_vec())
    {
    out.set_size(1,1);
    out[0] = eT(1);
    }
  else
    {
    const uword N = A.n_rows;
    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    const Row<eT> acc = sum(A);
    const Row<eT> sd  = stddev(A);

    out = (trans(A) * A);
    out -= (trans(acc) * acc)/eT(N);
    out /= norm_val;
    out /= trans(sd) * sd;
    }
  }



template<typename T>
inline
void
op_cor::direct_cor(Mat< std::complex<T> >& out, const Mat< std::complex<T> >& A, const uword norm_type)
  {
  arma_extra_debug_sigprint();

  typedef typename std::complex<T> eT;

  if(A.is_empty())
    {
    out.reset();
    return;
    }
  
  if(A.is_vec())
    {
    out.set_size(1,1);
    out[0] = eT(1);
    }
  else
    {
    const uword N = A.n_rows;
    const eT norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);

    const Row<eT> acc = sum(A);
    const Row<T>  sd  = stddev(A);

    out = trans(A) * A;               // out = strans(conj(A)) * A;
    out -= (trans(acc) * acc)/eT(N);  // out -= (strans(conj(acc)) * acc)/eT(N);
    out /= norm_val;

    //out = out / (trans(sd) * sd);
    out /= conv_to< Mat<eT> >::from(trans(sd) * sd);
    }
  }



template<typename T1>
inline
void
op_cor::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_cor>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const unwrap_check<T1> tmp(in.m, out);
  const Mat<eT>& A     = tmp.M;
  
  const uword norm_type = in.aux_uword_a;
  
  op_cor::direct_cor(out, A, norm_type);
  }



//! @}
