// Copyright (C) 2010-2012 Conrad Sanderson
// Copyright (C) 2010-2012 NICTA (www.nicta.com.au)
// Copyright (C) 2010 Dimitrios Bouzas
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_princomp
//! @{



//! \brief
//! principal component analysis -- 4 arguments version
//! coeff_out    -> principal component coefficients
//! score_out    -> projected samples
//! latent_out   -> eigenvalues of principal vectors
//! tsquared_out -> Hotelling's T^2 statistic
template<typename T1>
inline
bool
princomp
  (
         Mat<typename T1::elem_type>&    coeff_out,
         Mat<typename T1::elem_type>&    score_out,
         Col<typename T1::pod_type>&     latent_out,
         Col<typename T1::elem_type>&    tsquared_out,
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const bool status = op_princomp::direct_princomp(coeff_out, score_out, latent_out, tsquared_out, X);
  
  if(status == false)
    {
    coeff_out.reset();
    score_out.reset();
    latent_out.reset();
    tsquared_out.reset();
    
    arma_bad("princomp(): failed to converge", false);
    }
  
  return status;
  }



//! \brief
//! principal component analysis -- 3 arguments version
//! coeff_out    -> principal component coefficients
//! score_out    -> projected samples
//! latent_out   -> eigenvalues of principal vectors
template<typename T1>
inline
bool
princomp
  (
         Mat<typename T1::elem_type>&    coeff_out,
         Mat<typename T1::elem_type>&    score_out,
         Col<typename T1::pod_type>&     latent_out,
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const bool status = op_princomp::direct_princomp(coeff_out, score_out, latent_out, X); 
  
  if(status == false)
    {
    coeff_out.reset();
    score_out.reset();
    latent_out.reset();
    
    arma_bad("princomp(): failed to converge", false);
    }
  
  return status;
  }



//! \brief
//! principal component analysis -- 2 arguments version
//! coeff_out    -> principal component coefficients
//! score_out    -> projected samples
template<typename T1>
inline
bool
princomp
  (
         Mat<typename T1::elem_type>&    coeff_out,
         Mat<typename T1::elem_type>&    score_out,
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const bool status = op_princomp::direct_princomp(coeff_out, score_out, X); 
  
  if(status == false)
    {
    coeff_out.reset();
    score_out.reset();
    
    arma_bad("princomp(): failed to converge", false);
    }
  
  return status;
  }



//! \brief
//! principal component analysis -- 1 argument version
//! coeff_out    -> principal component coefficients
template<typename T1>
inline
bool
princomp
  (
         Mat<typename T1::elem_type>&    coeff_out,
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  const bool status = op_princomp::direct_princomp(coeff_out, X);
  
  if(status == false)
    {
    coeff_out.reset();
    
    arma_bad("princomp(): failed to converge", false);
    }
  
  return status;
  }



template<typename T1>
inline
const Op<T1, op_princomp>
princomp
  (
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = 0
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return Op<T1, op_princomp>(X.get_ref());
  }



//! @}
