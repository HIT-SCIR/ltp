// Copyright (C) 2014 Conrad Sanderson
// Copyright (C) 2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


namespace gmm_priv
{


// running_mean_scalar

template<typename eT>
class running_mean_scalar
  {
  public:
  
  inline running_mean_scalar();
  inline running_mean_scalar(const running_mean_scalar& in_rms);
  
  inline const running_mean_scalar& operator=(const running_mean_scalar& in_rms);
  
  arma_hot inline void operator() (const eT X);
  
  inline void  reset();
  
  inline uword count() const;
  inline eT    mean()  const;
  
  
  private:
  
  arma_aligned uword counter;
  arma_aligned eT    r_mean;
  };



// running_mean_vec

template<typename eT>
class running_mean_vec
  {
  public:
  
  inline running_mean_vec();
  inline running_mean_vec(const running_mean_vec& in_rmv);
  
  inline const running_mean_vec& operator=(const running_mean_vec& in_rmv);
  
  arma_hot inline void operator() (const Col<eT>& X, const uword index);
  
  inline void reset();
  
  inline uword          last_index() const;
  inline uword          count()      const;
  inline const Col<eT>& mean()       const;
  
  
  private:
  
  arma_aligned uword   last_i;
  arma_aligned uword   counter;
  arma_aligned Col<eT> r_mean;
  };



// distance

template<typename eT, uword dist_id>
struct distance {};


template<typename eT>
struct distance<eT, uword(1)>
  {
  arma_inline arma_hot static eT eval(const uword N, const eT* A, const eT* B, const eT*);
  };



template<typename eT>
struct distance<eT, uword(2)>
  {
  arma_inline arma_hot static eT eval(const uword N, const eT* A, const eT* B, const eT* C);
  };


}
