// Copyright (C) 2009-2013 Conrad Sanderson
// Copyright (C) 2009-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup running_stat_vec
//! @{


template<typename obj_type, bool> struct rsv_get_elem_type                  { };
template<typename obj_type>       struct rsv_get_elem_type<obj_type, false> { typedef          obj_type            elem_type; };
template<typename obj_type>       struct rsv_get_elem_type<obj_type, true>  { typedef typename obj_type::elem_type elem_type; };


//! Class for keeping statistics of a continuously sampled process / signal.
//! Useful if the storage of individual samples is not necessary or desired.
//! Also useful if the number of samples is not known beforehand or exceeds 
//! available memory.
template<typename obj_type>
class running_stat_vec
  {
  public:
  
  // voodoo for compatibility with old user code
  typedef typename rsv_get_elem_type<obj_type, is_Mat<obj_type>::value>::elem_type eT;
  
  typedef typename get_pod_type<eT>::result T;
  
  inline ~running_stat_vec();
  inline  running_stat_vec(const bool in_calc_cov = false);  // TODO: investigate char* overload, eg. "calc_cov", "no_calc_cov"
  
  inline running_stat_vec(const running_stat_vec& in_rsv);
  
  inline const running_stat_vec& operator=(const running_stat_vec& in_rsv);
  
  template<typename T1> arma_hot inline void operator() (const Base<              T, T1>& X);
  template<typename T1> arma_hot inline void operator() (const Base<std::complex<T>, T1>& X);
  
  inline void reset();
  
  inline const Mat<eT>&  mean() const;
  
  inline const Mat< T>&  var   (const uword norm_type = 0);
  inline       Mat< T>   stddev(const uword norm_type = 0) const;
  inline const Mat<eT>&  cov   (const uword norm_type = 0);
  
  inline const Mat<eT>& min() const;
  inline const Mat<eT>& max() const;
  
  inline T count() const;
  
  //
  //
  
  private:
  
  const bool calc_cov;
  
  arma_aligned arma_counter<T> counter;
  
  arma_aligned Mat<eT> r_mean;
  arma_aligned Mat< T> r_var;
  arma_aligned Mat<eT> r_cov;
  
  arma_aligned Mat<eT> min_val;
  arma_aligned Mat<eT> max_val;
  
  arma_aligned Mat< T> min_val_norm;
  arma_aligned Mat< T> max_val_norm;
  
  arma_aligned Mat< T> r_var_dummy;
  arma_aligned Mat<eT> r_cov_dummy;
  
  arma_aligned Mat<eT> tmp1;
  arma_aligned Mat<eT> tmp2;
  
  friend class running_stat_vec_aux;
  };



class running_stat_vec_aux
  {
  public:
  
  template<typename obj_type>
  inline static void
  update_stats
    (
    running_stat_vec<obj_type>& x,
    const                  Mat<typename running_stat_vec<obj_type>::eT>& sample,
    const typename arma_not_cx<typename running_stat_vec<obj_type>::eT>::result* junk = 0
    );
  
  template<typename obj_type>
  inline static void
  update_stats
    (
    running_stat_vec<obj_type>& x,
    const          Mat<std::complex< typename running_stat_vec<obj_type>::T > >& sample,
    const typename       arma_not_cx<typename running_stat_vec<obj_type>::eT>::result* junk = 0
    );
  
  template<typename obj_type>
  inline static void
  update_stats
    (
    running_stat_vec<obj_type>& x,
    const                  Mat< typename running_stat_vec<obj_type>::T >& sample,
    const typename arma_cx_only<typename running_stat_vec<obj_type>::eT>::result* junk = 0
    );
  
  template<typename obj_type>
  inline static void
  update_stats
    (
    running_stat_vec<obj_type>& x,
    const                   Mat<typename running_stat_vec<obj_type>::eT>& sample,
    const typename arma_cx_only<typename running_stat_vec<obj_type>::eT>::result* junk = 0
    );
  };



//! @}
