// Copyright (C) 2014 Conrad Sanderson
// Copyright (C) 2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


namespace gmm_priv
{


template<typename eT>
inline
gmm_diag<eT>::~gmm_diag()
  {
  arma_extra_debug_sigprint_this(this);
  
  arma_type_check(( (is_same_type<eT,float>::value == false) && (is_same_type<eT,double>::value == false) ));
  }



template<typename eT>
inline
gmm_diag<eT>::gmm_diag()
  {
  arma_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
gmm_diag<eT>::gmm_diag(const gmm_diag<eT>& x)
  {
  arma_extra_debug_sigprint_this(this);
  
  init(x);
  }



template<typename eT>
inline
const gmm_diag<eT>&
gmm_diag<eT>::operator=(const gmm_diag<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  init(x);
  
  return *this;
  }



template<typename eT>
inline
gmm_diag<eT>::gmm_diag(const uword in_n_dims, const uword in_n_gaus)
  {
  arma_extra_debug_sigprint_this(this);
  
  init(in_n_dims, in_n_gaus);
  }



template<typename eT>
inline
void
gmm_diag<eT>::reset()
  {
  arma_extra_debug_sigprint();
  
  mah_aux.reset();
  
  init(0, 0);
  }



template<typename eT>
inline
void
gmm_diag<eT>::reset(const uword in_n_dims, const uword in_n_gaus)
  {
  arma_extra_debug_sigprint();
  
  mah_aux.reset();
  
  init(in_n_dims, in_n_gaus);
  }



template<typename eT>
template<typename T1, typename T2, typename T3>
inline
void
gmm_diag<eT>::set_params(const Base<eT,T1>& in_means_expr, const Base<eT,T2>& in_dcovs_expr, const Base<eT,T3>& in_hefts_expr)
  {
  arma_extra_debug_sigprint();
  
  const unwrap<T1> tmp1(in_means_expr.get_ref());
  const unwrap<T2> tmp2(in_dcovs_expr.get_ref());
  const unwrap<T3> tmp3(in_hefts_expr.get_ref());
  
  const Mat<eT>& in_means = tmp1.M;
  const Mat<eT>& in_dcovs = tmp2.M;
  const Mat<eT>& in_hefts = tmp3.M;
  
  arma_debug_check
    (
    (size(in_means) != size(in_dcovs)) || (in_hefts.n_cols != in_means.n_cols) || (in_hefts.n_rows != 1),
    "gmm_diag::set_params(): given parameters have inconsistent and/or wrong sizes"
    );
  
  arma_debug_check( (in_means.is_finite() == false), "gmm_diag::set_params(): given means have non-finite values" );
  arma_debug_check( (in_dcovs.is_finite() == false), "gmm_diag::set_params(): given dcovs have non-finite values" );
  arma_debug_check( (in_hefts.is_finite() == false), "gmm_diag::set_params(): given hefts have non-finite values" );
  
  arma_debug_check( (any(vectorise(in_dcovs) <= eT(0))), "gmm_diag::set_params(): given dcovs have negative or zero values" );
  arma_debug_check( (any(vectorise(in_hefts) <  eT(0))), "gmm_diag::set_params(): given hefts have negative values"         );
  
  const eT s = accu(in_hefts);
  
  arma_debug_check( ((s < (eT(1) - Datum<eT>::eps)) || (s > (eT(1) + Datum<eT>::eps))), "gmm_diag::set_params(): sum of given hefts is not 1" );
  
  access::rw(means) = in_means;
  access::rw(dcovs) = in_dcovs;
  access::rw(hefts) = in_hefts;
  
  init_constants();
  }



template<typename eT>
template<typename T1>
inline
void
gmm_diag<eT>::set_means(const Base<eT,T1>& in_means_expr)
  {
  arma_extra_debug_sigprint();
  
  const unwrap<T1> tmp(in_means_expr.get_ref());
  
  const Mat<eT>& in_means = tmp.M;
  
  arma_debug_check( (size(in_means) != size(means)), "gmm_diag::set_means(): given means have incompatible size" );
  arma_debug_check( (in_means.is_finite() == false), "gmm_diag::set_means(): given means have non-finite values" );
  
  access::rw(means) = in_means;
  }



template<typename eT>
template<typename T1>
inline
void
gmm_diag<eT>::set_dcovs(const Base<eT,T1>& in_dcovs_expr)
  {
  arma_extra_debug_sigprint();
  
  const unwrap<T1> tmp(in_dcovs_expr.get_ref());
  
  const Mat<eT>& in_dcovs = tmp.M;
  
  arma_debug_check( (size(in_dcovs) != size(dcovs)),     "gmm_diag::set_dcovs(): given dcovs have incompatible size"       );
  arma_debug_check( (in_dcovs.is_finite() == false),     "gmm_diag::set_dcovs(): given dcovs have non-finite values"       );
  arma_debug_check( (any(vectorise(in_dcovs) <= eT(0))), "gmm_diag::set_dcovs(): given dcovs have negative or zero values" );
  
  access::rw(dcovs) = in_dcovs;
  
  init_constants();
  }



template<typename eT>
template<typename T1>
inline
void
gmm_diag<eT>::set_hefts(const Base<eT,T1>& in_hefts_expr)
  {
  arma_extra_debug_sigprint();
  
  const unwrap<T1> tmp(in_hefts_expr.get_ref());
  
  const Mat<eT>& in_hefts = tmp.M;
  
  arma_debug_check( (size(in_hefts) != size(hefts)),     "gmm_diag::set_hefts(): given hefts have incompatible size" );
  arma_debug_check( (in_hefts.is_finite() == false),     "gmm_diag::set_hefts(): given hefts have non-finite values" );
  arma_debug_check( (any(vectorise(in_hefts) <  eT(0))), "gmm_diag::set_hefts(): given hefts have negative values"   );
  
  const eT s = accu(in_hefts);
  
  arma_debug_check( ((s < (eT(1) - Datum<eT>::eps)) || (s > (eT(1) + Datum<eT>::eps))), "gmm_diag::set_hefts(): sum of given hefts is not 1" );
  
  access::rw(hefts) = in_hefts;
  
  log_hefts = log(hefts);  // TODO: possible issue when one of the hefts is zero
  }



template<typename eT>
inline
uword
gmm_diag<eT>::n_dims() const
  {
  return means.n_rows;
  }



template<typename eT>
inline
uword
gmm_diag<eT>::n_gaus() const
  {
  return means.n_cols;
  }



template<typename eT>
inline
bool
gmm_diag<eT>::load(const std::string name)
  {
  arma_extra_debug_sigprint();
  
  Cube<eT> Q;
  
  bool status = Q.load(name, arma_binary);
  
  if( (status == false) || (Q.n_slices != 2) )
    {
    reset();
    return false;
    }
  
  if( (Q.n_rows < 2) || (Q.n_cols < 1) )
    {
    reset();
    return true;
    }
  
  access::rw(hefts) = Q.slice(0).row(0);
  access::rw(means) = Q.slice(0).submat(1, 0, Q.n_rows-1, Q.n_cols-1);
  access::rw(dcovs) = Q.slice(1).submat(1, 0, Q.n_rows-1, Q.n_cols-1);
  
  init_constants();
  
  return true;
  }



template<typename eT>
inline
bool
gmm_diag<eT>::save(const std::string name) const
  {
  arma_extra_debug_sigprint();
  
  Cube<eT> Q(means.n_rows + 1, means.n_cols, 2);
  
  if(Q.n_elem > 0)
    {
    Q.slice(0).row(0) = hefts;
    Q.slice(1).row(0).zeros();  // reserved for future use
    
    Q.slice(0).submat(1, 0, size(means)) = means;
    Q.slice(1).submat(1, 0, size(dcovs)) = dcovs;
    }
  
  const bool status = Q.save(name, arma_binary);
  
  return status;
  }



template<typename eT>
inline
Col<eT>
gmm_diag<eT>::generate() const
  {
  arma_extra_debug_sigprint();
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  Col<eT> out( (N_gaus > 0) ? N_dims : uword(0) );
  
  if(N_gaus > 0)
    {
    const double val = randu<double>();
    
    double csum    = double(0);
    uword  gaus_id = 0;
    
    for(uword j=0; j < N_gaus; ++j)
      {
      csum += hefts[j];
      
      if(val <= csum)  { gaus_id = j; break; }
      }
    
    out =  randn< Col<eT> >(N_dims);    
    out %= sqrt(dcovs.col(gaus_id));
    out += means.col(gaus_id);
    }
  
  return out;
  }



template<typename eT>
inline
Mat<eT>
gmm_diag<eT>::generate(const uword N_vec) const
  {
  arma_extra_debug_sigprint();
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  Mat<eT> out( ( (N_gaus > 0) ? N_dims : uword(0) ), N_vec );
  
  if(N_gaus > 0)
    {
    const eT* hefts_mem = hefts.memptr();
    
    for(uword i=0; i < N_vec; ++i)
      {
      const double val = randu<double>();
      
      double csum    = double(0);
      uword  gaus_id = 0;
      
      for(uword j=0; j < N_gaus; ++j)
        {
        csum += hefts_mem[j];
        
        if(val <= csum)  { gaus_id = j; break; }
        }
      
      subview_col<eT> out_col = out.col(i);
      
      out_col =  randn< Col<eT> >(N_dims);    
      out_col %= sqrt(dcovs.col(gaus_id));
      out_col += means.col(gaus_id);
      }
    }
  
  return out;
  }



template<typename eT>
template<typename T1>
inline
eT
gmm_diag<eT>::log_p(const T1& expr, const gmm_empty_arg& junk1, typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == true))>::result* junk2) const
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  const quasi_unwrap<T1> tmp(expr);
  
  arma_debug_check( (tmp.M.n_rows != means.n_rows), "gmm_diag::log_p(): incompatible dimensions" );
  
  return internal_scalar_log_p( tmp.M.memptr() );
  }



template<typename eT>
template<typename T1>
inline
eT
gmm_diag<eT>::log_p(const T1& expr, const uword gaus_id, typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == true))>::result* junk2) const
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk2);
  
  const quasi_unwrap<T1> tmp(expr);
  
  arma_debug_check( (tmp.M.n_rows != means.n_rows), "gmm_diag::log_p(): incompatible dimensions" );
  
  arma_debug_check( (gaus_id >= means.n_cols), "gmm_diag::log_p(): specified gaussian is out of range" );
  
  return internal_scalar_log_p( tmp.M.memptr(), gaus_id );
  }



template<typename eT>
template<typename T1>
inline
Row<eT>
gmm_diag<eT>::log_p(const T1& expr, const gmm_empty_arg& junk1, typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == false))>::result* junk2) const
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  if(is_subview<T1>::value)
    {
    const subview<eT>& X = reinterpret_cast< const subview<eT>& >(expr);
    
    return internal_vec_log_p(X);
    }
  else
    {
    const unwrap<T1>   tmp(expr);
    const Mat<eT>& X = tmp.M;
    
    return internal_vec_log_p(X);
    }
  }



template<typename eT>
template<typename T1>
inline
Row<eT>
gmm_diag<eT>::log_p(const T1& expr, const uword gaus_id, typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == false))>::result* junk2) const
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk2);
  
  if(is_subview<T1>::value)
    {
    const subview<eT>& X = reinterpret_cast< const subview<eT>& >(expr);
    
    return internal_vec_log_p(X, gaus_id);
    }
  else
    {
    const unwrap<T1>   tmp(expr);
    const Mat<eT>& X = tmp.M;
    
    return internal_vec_log_p(X, gaus_id);
    }
  }



template<typename eT>
template<typename T1>
inline
eT
gmm_diag<eT>::avg_log_p(const Base<eT,T1>& expr) const
  {
  arma_extra_debug_sigprint();
  
  if(is_subview<T1>::value)
    {
    const subview<eT>& X = reinterpret_cast< const subview<eT>& >( expr.get_ref() );
    
    return internal_avg_log_p(X);
    }
  else
    {
    const unwrap<T1>   tmp(expr.get_ref());
    const Mat<eT>& X = tmp.M;
    
    return internal_avg_log_p(X);
    }
  }



template<typename eT>
template<typename T1>
inline
eT
gmm_diag<eT>::avg_log_p(const Base<eT,T1>& expr, const uword gaus_id) const
  {
  arma_extra_debug_sigprint();
  
  if(is_subview<T1>::value)
    {
    const subview<eT>& X = reinterpret_cast< const subview<eT>& >( expr.get_ref() );
    
    return internal_avg_log_p(X, gaus_id);
    }
  else
    {
    const unwrap<T1>   tmp(expr.get_ref());
    const Mat<eT>& X = tmp.M;
    
    return internal_avg_log_p(X, gaus_id);
    }
  }



template<typename eT>
template<typename T1>
inline
uword
gmm_diag<eT>::assign(const T1& expr, const gmm_dist_mode& dist, typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == true))>::result* junk) const
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  if(is_subview_col<T1>::value)
    {
    const subview_col<eT>& X = reinterpret_cast< const subview_col<eT>& >(expr);
    
    return internal_scalar_assign(X, dist);
    }
  else
    {
    const unwrap<T1>   tmp(expr);
    const Mat<eT>& X = tmp.M;
    
    return internal_scalar_assign(X, dist);
    }
  }



template<typename eT>
template<typename T1>
inline
urowvec
gmm_diag<eT>::assign(const T1& expr, const gmm_dist_mode& dist, typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == false))>::result* junk) const
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  urowvec out;
  
  if(is_subview<T1>::value)
    {
    const subview<eT>& X = reinterpret_cast< const subview<eT>& >(expr);
    
    internal_vec_assign(out, X, dist);
    }
  else
    {
    const unwrap<T1>   tmp(expr);
    const Mat<eT>& X = tmp.M;
    
    internal_vec_assign(out, X, dist);
    }
  
  return out;
  }



template<typename eT>
template<typename T1>
inline
urowvec
gmm_diag<eT>::raw_hist(const Base<eT,T1>& expr, const gmm_dist_mode& dist_mode) const
  {
  arma_extra_debug_sigprint();
  
  const unwrap<T1>   tmp(expr.get_ref());
  const Mat<eT>& X = tmp.M;
  
  arma_debug_check( (X.n_rows != means.n_rows), "gmm_diag::raw_hist(): incompatible dimensions" );
  
  arma_debug_check( ((dist_mode != eucl_dist) && (dist_mode != prob_dist)), "gmm_diag::raw_hist(): unsupported distance mode" );
  
  urowvec hist;
  
  internal_raw_hist(hist, X, dist_mode);
  
  return hist;
  }



template<typename eT>
template<typename T1>
inline
Row<eT>
gmm_diag<eT>::norm_hist(const Base<eT,T1>& expr, const gmm_dist_mode& dist_mode) const
  {
  arma_extra_debug_sigprint();
  
  const unwrap<T1>   tmp(expr.get_ref());
  const Mat<eT>& X = tmp.M;
  
  arma_debug_check( (X.n_rows != means.n_rows), "gmm_diag::norm_hist(): incompatible dimensions" );
  
  arma_debug_check( ((dist_mode != eucl_dist) && (dist_mode != prob_dist)), "gmm_diag::norm_hist(): unsupported distance mode" );
  
  urowvec hist;
  
  internal_raw_hist(hist, X, dist_mode);
  
  const uword  hist_n_elem = hist.n_elem;
  const uword* hist_mem    = hist.memptr();
  
  eT acc = eT(0);
  for(uword i=0; i<hist_n_elem; ++i)  { acc += eT(hist_mem[i]); }
  
  if(acc == eT(0))  { acc = eT(1); }
  
  Row<eT> out(hist_n_elem);
  
  eT* out_mem = out.memptr();
  
  for(uword i=0; i<hist_n_elem; ++i)  { out_mem[i] = eT(hist_mem[i]) / acc; }
  
  return out;
  }



template<typename eT>
template<typename T1>
inline
bool
gmm_diag<eT>::learn
  (
  const Base<eT,T1>&   data,
  const uword          N_gaus,
  const gmm_dist_mode& dist_mode,
  const gmm_seed_mode& seed_mode,
  const uword          km_iter,
  const uword          em_iter,
  const eT             var_floor,
  const bool           print_mode
  )
  {
  arma_extra_debug_sigprint();
  
  const bool dist_mode_ok = (dist_mode == eucl_dist) || (dist_mode == maha_dist);
  
  const bool seed_mode_ok = \
       (seed_mode == keep_existing)
    || (seed_mode == static_subset)
    || (seed_mode == static_spread)
    || (seed_mode == random_subset)
    || (seed_mode == random_spread);
  
  arma_debug_check( (dist_mode_ok == false), "gmm_diag::learn(): dist_mode must be eucl_dist or maha_dist" );
  arma_debug_check( (seed_mode_ok == false), "gmm_diag::learn(): unknown seed_mode"                        );
  arma_debug_check( (var_floor < eT(0)    ), "gmm_diag::learn(): variance floor is negative"               );
  
  const unwrap<T1>   tmp_X(data.get_ref());
  const Mat<eT>& X = tmp_X.M;
  
  if(X.is_empty()          )  { arma_warn(true, "gmm_diag::learn(): given matrix is empty"             ); return false; }
  if(X.is_finite() == false)  { arma_warn(true, "gmm_diag::learn(): given matrix has non-finite values"); return false; }
  
  if(N_gaus == 0)  { reset(); return true; }
  
  if(dist_mode == maha_dist)
    {
    mah_aux = var(X,1,1);
    
    const uword mah_aux_n_elem = mah_aux.n_elem;
          eT*   mah_aux_mem    = mah_aux.memptr();
    
    for(uword i=0; i < mah_aux_n_elem; ++i)
      {
      const eT val = mah_aux_mem[i];
      
      mah_aux_mem[i] = ((val != eT(0)) && arma_isfinite(val)) ? eT(1) / val : eT(1);
      }
    }
  
  
  // copy current model, in case of failure by k-means and/or EM
  
  const gmm_diag<eT> orig = (*this);
  
  
  // initial means
  
  if(seed_mode == keep_existing)
    {
    if(means.is_empty()        )  { arma_warn(true, "gmm_diag::learn(): no existing means"      ); return false; }
    if(X.n_rows != means.n_rows)  { arma_warn(true, "gmm_diag::learn(): dimensionality mismatch"); return false; }
    
    // TODO: also check for number of vectors?
    }
  else
    {
    if(X.n_cols < N_gaus)  { arma_warn(true, "gmm_diag::learn(): number of vectors is less than number of gaussians"); return false; }
    
    reset(X.n_rows, N_gaus);
    
    if(print_mode)  { get_stream_err2() << "gmm_diag::learn(): generating initial means\n"; }
    
         if(dist_mode == eucl_dist)  { generate_initial_means<1>(X, seed_mode); }
    else if(dist_mode == maha_dist)  { generate_initial_means<2>(X, seed_mode); }
    }
  
  
  // k-means
  
  if(km_iter > 0)
    {
    const arma_ostream_state stream_state(get_stream_err2());
    
    bool status = false;
    
         if(dist_mode == eucl_dist)  { status = km_iterate<1>(X, km_iter, print_mode); }
    else if(dist_mode == maha_dist)  { status = km_iterate<2>(X, km_iter, print_mode); }
    
    stream_state.restore(get_stream_err2());
    
    if(status == false)  { arma_warn(true, "gmm_diag::learn(): k-means algorithm failed"); init(orig); return false; }
    }
  
  
  // initial dcovs
  
  const eT vfloor = (eT(var_floor) > eT(0)) ? eT(var_floor) : std::numeric_limits<eT>::min();
  
  if(seed_mode != keep_existing)
    {
    if(print_mode)  { get_stream_err2() << "gmm_diag::learn(): generating initial covariances\n"; }
    
         if(dist_mode == eucl_dist)  { generate_initial_dcovs_and_hefts<1>(X, vfloor); }
    else if(dist_mode == maha_dist)  { generate_initial_dcovs_and_hefts<2>(X, vfloor); }
    }
  
  
  // EM algorithm
  
  if(em_iter > 0)
    {
    const arma_ostream_state stream_state(get_stream_err2());
    
    const bool status = em_iterate(X, em_iter, vfloor, print_mode);
    
    stream_state.restore(get_stream_err2());
    
    if(status == false)  { arma_warn(true, "gmm_diag::learn(): EM algorithm failed"); init(orig); return false; }
    }
  
  mah_aux.reset();
  
  init_constants();
  
  return true;
  }



//
//
//



template<typename eT>
inline
void
gmm_diag<eT>::init(const gmm_diag<eT>& x)
  {
  arma_extra_debug_sigprint();
  
  gmm_diag<eT>& t = *this;
  
  if(&t != &x)
    {
    access::rw(t.means) = x.means;
    access::rw(t.dcovs) = x.dcovs;
    access::rw(t.hefts) = x.hefts;
    
    init_constants();
    }
  }



template<typename eT>
inline
void
gmm_diag<eT>::init(const uword in_n_dims, const uword in_n_gaus)
  {
  arma_extra_debug_sigprint();
  
  access::rw(means).zeros(in_n_dims, in_n_gaus);
  
  access::rw(dcovs).ones(in_n_dims, in_n_gaus);
  
  access::rw(hefts).set_size(in_n_gaus);
  
  access::rw(hefts).fill(eT(1) / eT(in_n_gaus));
  
  init_constants();
  }



template<typename eT>
inline
void
gmm_diag<eT>::init_constants()
  {
  arma_extra_debug_sigprint();
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  const eT tmp = (eT(N_dims)/eT(2)) * std::log(eT(2) * Datum<eT>::pi);
  
  log_det_etc.set_size(N_gaus);
  
  for(uword i=0; i<N_gaus; ++i)
    {
    const eT logdet = accu( log(dcovs.col(i)) );
    
    log_det_etc[i] = eT(-1) * ( tmp + eT(0.5) * logdet );
    }
  
  log_hefts = log(hefts);  // TODO: possible issue when one of the hefts is zero
  }



template<typename eT>
inline
umat
gmm_diag<eT>::internal_gen_boundaries(const uword N) const
  {
  arma_extra_debug_sigprint();
  
  #if defined(_OPENMP)
    // const uword n_cores = 0;
    const uword n_cores   = uword(omp_get_num_procs());
    const uword n_threads = (n_cores > 0) ? ( (n_cores <= N) ? n_cores : 1 ) : 1;
  #else
    // static const uword n_cores   = 0;
    static const uword n_threads = 1;
  #endif
  
  // get_stream_err2() << "gmm_diag::internal_gen_boundaries(): n_cores:   " << n_cores   << '\n';
  // get_stream_err2() << "gmm_diag::internal_gen_boundaries(): n_threads: " << n_threads << '\n';
  
  umat boundaries(2, n_threads);
  
  if(N > 0)
    {
    const uword chunk_size = N / n_threads;
    
    uword count = 0;
    
    for(uword t=0; t<n_threads; t++)
      {
      boundaries.at(0,t) = count;
      
      count += chunk_size;
      
      boundaries.at(1,t) = count-1;
      }
    
    boundaries.at(1,n_threads-1) = N - 1;
    }
  else
    {
    boundaries.zeros();
    }
  
  // get_stream_err2() << "gmm_diag::internal_gen_boundaries(): boundaries: " << '\n' << boundaries << '\n';
  
  return boundaries;
  }



template<typename eT>
arma_hot
inline
eT
gmm_diag<eT>::internal_scalar_log_p(const eT* x) const
  {
  arma_extra_debug_sigprint();
  
  const eT* log_hefts_mem = log_hefts.mem;
  
  const uword N_gaus = means.n_cols;
  
  if(N_gaus > 0)
    {
    eT log_sum = internal_scalar_log_p(x, 0) + log_hefts_mem[0];
    
    for(uword g=1; g < N_gaus; ++g)
      {
      const eT tmp = internal_scalar_log_p(x, g) + log_hefts_mem[g];
      
      log_sum = log_add_exp(log_sum, tmp);
      }
    
    return log_sum;
    }
  else
    {
    return -Datum<eT>::inf;
    }
  }



template<typename eT>
arma_hot
inline
eT
gmm_diag<eT>::internal_scalar_log_p(const eT* x, const uword g) const
  {
  arma_extra_debug_sigprint();
  
  const eT* mean = means.colptr(g);
  const eT* dcov = dcovs.colptr(g);
  
  const uword N_dims = means.n_rows;
  
  eT val_i = eT(0);
  eT val_j = eT(0);
  
  uword i,j;
  
  for(i=0, j=1; j<N_dims; i+=2, j+=2)
    {
    eT tmp_i = x[i];
    eT tmp_j = x[j];
    
    tmp_i -= mean[i];
    tmp_j -= mean[j];
    
    val_i += (tmp_i*tmp_i) / dcov[i];
    val_j += (tmp_j*tmp_j) / dcov[j];
    }
  
  if(i < N_dims)
    {
    const eT tmp = x[i] - mean[i];
    
    val_i += (tmp*tmp) / dcov[i];
    }
  
  return eT(-0.5)*(val_i + val_j) + log_det_etc.mem[g];
  }



template<typename eT>
template<typename T1>
inline
Row<eT>
gmm_diag<eT>::internal_vec_log_p(const T1& X) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (X.n_rows != means.n_rows), "gmm_diag::log_p(): incompatible dimensions" );
  
  const uword N = X.n_cols;
  
  Row<eT> out(N);
  
  if(N > 0)
    {
    #if defined(_OPENMP)
      {
      const arma_omp_state save_omp_state;
      
      const umat boundaries = internal_gen_boundaries(N);
      
      const uword n_threads = boundaries.n_cols;
      
      #pragma omp parallel for
      for(uword t=0; t < n_threads; ++t)
        {
        const uword start_index = boundaries.at(0,t);
        const uword   end_index = boundaries.at(1,t);
        
        eT* out_mem = out.memptr();
        
        for(uword i=start_index; i <= end_index; ++i)
          {
          out_mem[i] = internal_scalar_log_p( X.colptr(i) );
          }
        }
      }
    #else
      {
      eT* out_mem = out.memptr();
      
      for(uword i=0; i < N; ++i)
        {
        out_mem[i] = internal_scalar_log_p( X.colptr(i) );
        }
      }
    #endif
    }
  
  return out;
  }



template<typename eT>
template<typename T1>
inline
Row<eT>
gmm_diag<eT>::internal_vec_log_p(const T1& X, const uword gaus_id) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (X.n_rows != means.n_rows), "gmm_diag::log_p(): incompatible dimensions" );
  arma_debug_check( (gaus_id  >= means.n_cols), "gmm_diag::log_p(): gaus_id is out of range" );
  
  const uword N = X.n_cols;
  
  Row<eT> out(N);
  
  if(N > 0)
    {
    #if defined(_OPENMP)
      {
      const arma_omp_state save_omp_state;
      
      const umat boundaries = internal_gen_boundaries(N);
      
      const uword n_threads = boundaries.n_cols;
      
      #pragma omp parallel for
      for(uword t=0; t < n_threads; ++t)
        {
        const uword start_index = boundaries.at(0,t);
        const uword   end_index = boundaries.at(1,t);
        
        eT* out_mem = out.memptr();
        
        for(uword i=start_index; i <= end_index; ++i)
          {
          out_mem[i] = internal_scalar_log_p( X.colptr(i), gaus_id );
          }
        }
      }
    #else
      {
      eT* out_mem = out.memptr();
      
      for(uword i=0; i < N; ++i)
        {
        out_mem[i] = internal_scalar_log_p( X.colptr(i), gaus_id );
        }
      }
    #endif
    }
  
  return out;
  }



template<typename eT>
template<typename T1>
inline
eT
gmm_diag<eT>::internal_avg_log_p(const T1& X) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (X.n_rows != means.n_rows), "gmm_diag::avg_log_p(): incompatible dimensions" );
    
  const uword N = X.n_cols;
  
  if(N == 0)  { return (-Datum<eT>::inf); }
  
  
  #if defined(_OPENMP)
    {
    const arma_omp_state save_omp_state;
    
    const umat boundaries = internal_gen_boundaries(N);
    
    const uword n_threads = boundaries.n_cols;
    
    field< running_mean_scalar<eT> > t_running_means(n_threads);
    
    
    #pragma omp parallel for
    for(uword t=0; t < n_threads; ++t)
      {
      const uword start_index = boundaries.at(0,t);
      const uword   end_index = boundaries.at(1,t);
      
      running_mean_scalar<eT>& current_running_mean = t_running_means[t];
      
      for(uword i=start_index; i <= end_index; ++i)
        {
        current_running_mean( internal_scalar_log_p( X.colptr(i) ) );
        }
      }
    
    
    eT avg = eT(0);
    
    for(uword t=0; t < n_threads; ++t)
      {
      running_mean_scalar<eT>& current_running_mean = t_running_means[t];
      
      const eT w = eT(current_running_mean.count()) / eT(N);
      
      avg += w * current_running_mean.mean();
      }
    
    return avg;
    }
  #else
    {
    running_mean_scalar<eT> running_mean;
    
    for(uword i=0; i<N; ++i)
      {
      running_mean( internal_scalar_log_p( X.colptr(i) ) );
      }
    
    return running_mean.mean();
    }
  #endif
  }



template<typename eT>
template<typename T1>
inline
eT
gmm_diag<eT>::internal_avg_log_p(const T1& X, const uword gaus_id) const
  {
  arma_extra_debug_sigprint();
  
  arma_debug_check( (X.n_rows != means.n_rows), "gmm_diag::avg_log_p(): incompatible dimensions" );
  arma_debug_check( (gaus_id  >= means.n_cols), "gmm_diag::avg_log_p(): specified gaussian is out of range"    );
  
  const uword N = X.n_cols;
  
  if(N == 0)  { return (-Datum<eT>::inf); }
  
  
  #if defined(_OPENMP)
    {
    const arma_omp_state save_omp_state;
    
    const umat boundaries = internal_gen_boundaries(N);
    
    const uword n_threads = boundaries.n_cols;
    
    field< running_mean_scalar<eT> > t_running_means(n_threads);
    
    
    #pragma omp parallel for
    for(uword t=0; t < n_threads; ++t)
      {
      const uword start_index = boundaries.at(0,t);
      const uword   end_index = boundaries.at(1,t);
      
      running_mean_scalar<eT>& current_running_mean = t_running_means[t];
      
      for(uword i=start_index; i <= end_index; ++i)
        {
        current_running_mean( internal_scalar_log_p( X.colptr(i), gaus_id) );
        }
      }
    
    
    eT avg = eT(0);
    
    for(uword t=0; t < n_threads; ++t)
      {
      running_mean_scalar<eT>& current_running_mean = t_running_means[t];
      
      const eT w = eT(current_running_mean.count()) / eT(N);
      
      avg += w * current_running_mean.mean();
      }
    
    return avg;
    }
  #else
    {
    running_mean_scalar<eT> running_mean;
    
    for(uword i=0; i<N; ++i)
      {
      running_mean( internal_scalar_log_p( X.colptr(i), gaus_id ) );
      }
    
    return running_mean.mean();
    }
  #endif
  }



template<typename eT>
template<typename T1>
inline
uword
gmm_diag<eT>::internal_scalar_assign(const T1& X, const gmm_dist_mode& dist_mode) const
  {
  arma_extra_debug_sigprint();
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  arma_debug_check( (X.n_rows != N_dims), "gmm_diag::assign(): incompatible dimensions" );
  arma_debug_check( (N_gaus == 0),        "gmm_diag::assign(): model has no means"      );
  
  const eT* X_mem = X.colptr(0);
  
  if(dist_mode == eucl_dist)
    {
    eT    best_dist = Datum<eT>::inf;
    uword best_g    = 0;
    
    for(uword g=0; g < N_gaus; ++g)
      {
      const eT tmp_dist = distance<eT,1>::eval(N_dims, X_mem, means.colptr(g), X_mem);
      
      if(tmp_dist <= best_dist)
        {
        best_dist = tmp_dist;
        best_g    = g;
        }
      }
    
    return best_g;
    }
  else
  if(dist_mode == prob_dist)
    {
    const eT* log_hefts_mem = log_hefts.memptr();
    
    eT    best_p = -Datum<eT>::inf;
    uword best_g = 0;
    
    for(uword g=0; g < N_gaus; ++g)
      {
      const eT tmp_p = internal_scalar_log_p(X_mem, g) + log_hefts_mem[g];
      
      if(tmp_p >= best_p)
        {
        best_p = tmp_p;
        best_g = g;
        }
      }
    
    return best_g;
    }
  else
    {
    arma_debug_check(true, "gmm_diag::assign(): unsupported distance mode");
    }
  
  return uword(0);
  }



template<typename eT>
template<typename T1>
inline
void
gmm_diag<eT>::internal_vec_assign(urowvec& out, const T1& X, const gmm_dist_mode& dist_mode) const
  {
  arma_extra_debug_sigprint();
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  arma_debug_check( (X.n_rows != N_dims), "gmm_diag::assign(): incompatible dimensions" );
  
  const uword X_n_cols = (N_gaus > 0) ? X.n_cols : 0;
  
  out.set_size(1,X_n_cols);
  
  uword* out_mem = out.memptr();
  
  if(dist_mode == eucl_dist)
    {
    for(uword i=0; i<X_n_cols; ++i)
      {
      const eT* X_colptr = X.colptr(i);
       
      eT    best_dist = Datum<eT>::inf;
      uword best_g    = 0;
      
      for(uword g=0; g<N_gaus; ++g)
        {
        const eT tmp_dist = distance<eT,1>::eval(N_dims, X_colptr, means.colptr(g), X_colptr);
        
        if(tmp_dist <= best_dist)
          {
          best_dist = tmp_dist;
          best_g    = g;
          }
        }
      
      out_mem[i] = best_g;
      }
    }
  else
  if(dist_mode == prob_dist)
    {
    const eT* log_hefts_mem = log_hefts.memptr();
    
    for(uword i=0; i<X_n_cols; ++i)
      {
      const eT* X_colptr = X.colptr(i);
       
      eT    best_p = -Datum<eT>::inf;
      uword best_g = 0;
      
      for(uword g=0; g<N_gaus; ++g)
        {
        const eT tmp_p = internal_scalar_log_p(X_colptr, g) + log_hefts_mem[g];
        
        if(tmp_p >= best_p)
          {
          best_p = tmp_p;
          best_g = g;
          }
        }
      
      out_mem[i] = best_g;
      }
    }
  else
    {
    arma_debug_check(true, "gmm_diag::assign(): unsupported distance mode");
    }
  }




template<typename eT>
inline
void
gmm_diag<eT>::internal_raw_hist(urowvec& hist, const Mat<eT>& X, const gmm_dist_mode& dist_mode) const
  {
  arma_extra_debug_sigprint();
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  const uword X_n_cols = X.n_cols;
  
  hist.zeros(N_gaus);
  
  if(N_gaus == 0)  { return; }
  
  uword* hist_mem = hist.memptr();
  
  if(dist_mode == eucl_dist)
    {
    for(uword i=0; i<X_n_cols; ++i)
      {
      const eT* X_colptr = X.colptr(i);
       
      eT    best_dist = Datum<eT>::inf;
      uword best_g    = 0;
      
      for(uword g=0; g < N_gaus; ++g)
        {
        const eT tmp_dist = distance<eT,1>::eval(N_dims, X_colptr, means.colptr(g), X_colptr);
        
        if(tmp_dist <= best_dist)
          {
          best_dist = tmp_dist;
          best_g    = g;
          }
        }
      
      hist_mem[best_g]++;
      }
    }
  else
  if(dist_mode == prob_dist)
    {
    const eT* log_hefts_mem = log_hefts.memptr();
    
    for(uword i=0; i<X_n_cols; ++i)
      {
      const eT* X_colptr = X.colptr(i);
        
      eT    best_p = -Datum<eT>::inf;
      uword best_g = 0;
      
      for(uword g=0; g < N_gaus; ++g)
        {
        const eT tmp_p = internal_scalar_log_p(X_colptr, g) + log_hefts_mem[g];
        
        if(tmp_p >= best_p)
          {
          best_p = tmp_p;
          best_g = g;
          }
        }
      
      hist_mem[best_g]++;
      }
    }
  }



template<typename eT>
template<uword dist_id>
inline
void
gmm_diag<eT>::generate_initial_means(const Mat<eT>& X, const gmm_seed_mode& seed_mode)
  {
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  if( (seed_mode == static_subset) || (seed_mode == random_subset) )
    {
    uvec initial_indices;
    
         if(seed_mode == static_subset)  { initial_indices = linspace<uvec>(0, X.n_cols-1, N_gaus);                   }
    else if(seed_mode == random_subset)  { initial_indices = uvec(sort_index(randu<vec>(X.n_cols))).rows(0,N_gaus-1); }
    
    // not using randi() here as on some primitive systems it produces vectors with non-unique values
    
    // initial_indices.print("initial_indices:");
    
    access::rw(means) = X.cols(initial_indices);
    }
  else
  if( (seed_mode == static_spread) || (seed_mode == random_spread) )
    {
    uword start_index = 0;
    
         if(seed_mode == static_spread)  { start_index = X.n_cols / 2;                                         }
    else if(seed_mode == random_spread)  { start_index = as_scalar(randi<uvec>(1, distr_param(0,X.n_cols-1))); }
    
    access::rw(means).col(0) = X.unsafe_col(start_index);
    
    const eT* mah_aux_mem = mah_aux.memptr();
    
    running_stat<double> rs;
    
    for(uword g=1; g < N_gaus; ++g)
      {
      eT    max_dist = eT(0);
      uword best_i   = uword(0);
      
      for(uword i=0; i < X.n_cols; ++i)
        {
        rs.reset();
        
        const eT* X_colptr = X.colptr(i);
        
        bool ignore_i = false;
        
        // find the average distance between sample i and the means so far
        for(uword h = 0; h < g; ++h)
          {
          const eT dist = distance<eT,dist_id>::eval(N_dims, X_colptr, means.colptr(h), mah_aux_mem);
          
          // ignore sample already selected as a mean
          if(dist == eT(0))  { ignore_i = true; break; }
          else               { rs(dist);               }
          }
        
        if( (rs.mean() >= max_dist) && (ignore_i == false))
          {
          max_dist = rs.mean(); best_i = i;
          }
        }
      
      // set the mean to the sample that is the furthest away from the means so far
      access::rw(means).col(g) = X.unsafe_col(best_i);
      }
    }
  
  // get_stream_err2() << "generate_initial_means():" << '\n';
  // means.print();
  }



template<typename eT>
template<uword dist_id>
inline
void
gmm_diag<eT>::generate_initial_dcovs_and_hefts(const Mat<eT>& X, const eT var_floor)
  {
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  field< running_stat_vec< Col<eT> > > rs(N_gaus);
  
  const eT* mah_aux_mem = mah_aux.memptr();
  
  for(uword i=0; i<X.n_cols; ++i)
    {
    const eT* X_colptr = X.colptr(i);
    
    double min_dist = Datum<eT>::inf;
    uword  best_g   = 0;
    
    for(uword g=0; g<N_gaus; ++g)
      {
      const double dist = distance<eT,dist_id>::eval(N_dims, X_colptr, means.colptr(g), mah_aux_mem);
      
      if(dist <= min_dist)  { min_dist = dist; best_g = g; }
      }
    
    rs(best_g)(X.unsafe_col(i));
    }
  
  for(uword g=0; g<N_gaus; ++g)
    {
    if( rs(g).count() >= eT(2) )
      {
      access::rw(dcovs).col(g) = rs(g).var(1);
      }
    else
      {
      access::rw(dcovs).col(g).ones();
      }
    
    access::rw(hefts)(g) = (std::max)(eT(1), rs(g).count()) / eT(X.n_cols);
    }
  
  em_fix_params(var_floor);
  }



//! multi-threaded implementation of k-means, inspired by MapReduce
template<typename eT>
template<uword dist_id>
inline
bool
gmm_diag<eT>::km_iterate(const Mat<eT>& X, const uword max_iter, const bool verbose)
  {
  arma_extra_debug_sigprint();
  
  if(verbose)
    {
    get_stream_err2().unsetf(ios::showbase);
    get_stream_err2().unsetf(ios::uppercase);
    get_stream_err2().unsetf(ios::showpos);
    get_stream_err2().unsetf(ios::scientific);
    
    get_stream_err2().setf(ios::right);
    get_stream_err2().setf(ios::fixed);
    }
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  Mat<eT> old_means = means;
  Mat<eT> new_means = means;
  
  running_mean_scalar<double> rs_delta;
  
  field< running_mean_vec<eT> > running_means(N_gaus);
  
  const eT* mah_aux_mem = mah_aux.memptr();
  
  
  #if defined(_OPENMP)
    const arma_omp_state save_omp_state;
    
    const umat boundaries = internal_gen_boundaries(X.n_cols);
    
    const uword n_threads = boundaries.n_cols;
    
    field< field< running_mean_vec<eT> > > t_running_means(n_threads);
    
    for(uword t=0; t < n_threads; ++t)  { t_running_means[t].set_size(N_gaus); }
    
    vec tmp_mean(N_dims);
    
    if(verbose)
      {
      get_stream_err2() << "gmm_diag::learn(): k-means: n_threads: " << n_threads  << '\n';
      }
  #endif
  
  
  for(uword iter=1; iter <= max_iter; ++iter)
    {
    #if defined(_OPENMP)
      {
      for(uword t=0; t < n_threads; ++t)
        {
        for(uword g=0; g < N_gaus; ++g)  { t_running_means[t][g].reset(); }
        }
      
      
      // km_update_stats() is the "map" operation, which produces partial means
      
      #pragma omp parallel for
      for(uword t=0; t < n_threads; ++t)
        {
        field< running_mean_vec<eT> >& current_running_means = t_running_means[t];
        
        km_update_stats<dist_id>(X, boundaries.at(0,t), boundaries.at(1,t), old_means, current_running_means);
        }
      
      
      // the "reduce" operation, which combines the partial means produced by the separate threads;
      // takes into account the counts for each mean
      
      for(uword g=0; g < N_gaus; ++g)
        {
        uword total_count = 0;
        
        for(uword t=0; t < n_threads; ++t)  { total_count += t_running_means[t][g].count(); }
        
        tmp_mean.zeros();
        
        bool  dead       = true;
        uword last_index = 0;
        
        if(total_count > 0)
          {
          for(uword t=0; t < n_threads; ++t)
            {
            const eT w = eT(t_running_means[t][g].count()) / eT(total_count);
            
            if(w > eT(0))
              {
              tmp_mean += w * t_running_means[t][g].mean();
              
              dead       = false;
              last_index = t_running_means[t][g].last_index();
              }
            }
          }
        
        running_means[g].reset();
        
        if(dead == false)  { running_means[g](tmp_mean, last_index); }
        }
      }
    #else
      {
      for(uword g=0; g < N_gaus; ++g)  { running_means[g].reset(); }
      
      km_update_stats<dist_id>(X, 0, X.n_cols-1, old_means, running_means);
      }
    #endif
    
    uword n_dead_means = 0;
    
    for(uword g=0; g < N_gaus; ++g)
      {
      if(running_means[g].count() > 0)
        {
        new_means.col(g) = running_means[g].mean();
        }
      else
        {
        n_dead_means++;
        }
      }
    
    if(n_dead_means > 0)
      {
      if(verbose)  { get_stream_err2() << "gmm_diag::learn(): k-means: recovering from dead means\n"; }
      
      if(n_dead_means == 1)
        {
        uword dead_g         = 0;
        uword populous_g     = 0;
        uword populous_count = running_means(0).count(); 
        
        for(uword g=1; g < N_gaus; ++g)
          {
          const uword count = running_means(g).count();
          
          if(count == 0)  { dead_g = g; }
          
          if(populous_count < count)
            {
            populous_count = count;
            populous_g     = g;
            }
          }
        
        if( (populous_count <= 2) || (dead_g == populous_g) )  { return false; }
        
        new_means.col(dead_g) = X.unsafe_col( running_means(populous_g).last_index() );
        }
      else
        {
        uword dead_g = 0;
        
        for(uword live_g = 0; live_g < N_gaus; ++live_g)
          {
          if(running_means(live_g).count() >= 2)
            {
            for(; dead_g < N_gaus; ++dead_g)
              {
              if(running_means(dead_g).count() == 0)  { break; }
              }
            
            new_means.col(dead_g) = X.unsafe_col( running_means(live_g).last_index() );
            
            dead_g++;
            }
          }
        }
      }
    
    rs_delta.reset();
    
    for(uword g=0; g < N_gaus; ++g)
      {
      rs_delta( distance<eT,dist_id>::eval(N_dims, old_means.colptr(g), new_means.colptr(g), mah_aux_mem) );
      }
    
    if(verbose)
      {
      get_stream_err2() << "gmm_diag::learn(): k-means: iteration: ";
      get_stream_err2().unsetf(ios::scientific);
      get_stream_err2().setf(ios::fixed);
      get_stream_err2().width(std::streamsize(4));
      get_stream_err2() << iter;
      get_stream_err2() << "   delta: ";
      get_stream_err2().unsetf(ios::fixed);
      //get_stream_err2().setf(ios::scientific);
      get_stream_err2() << rs_delta.mean() << '\n';
      }
    
    arma::swap(old_means, new_means);
    
    if(rs_delta.mean() <= Datum<eT>::eps)  { break; }
    }
  
  access::rw(means) = old_means;
  
  return true;
  }



template<typename eT>
template<uword dist_id>
inline
void
gmm_diag<eT>::km_update_stats(const Mat<eT>& X, const uword start_index, const uword end_index, const Mat<eT>& old_means, field< running_mean_vec<eT> >& running_means) const
  {
  arma_extra_debug_sigprint();
  
  // get_stream_err2() << "km_update_stats(): start_index: " << start_index << '\n';
  // get_stream_err2() << "km_update_stats():   end_index: " <<   end_index << '\n';
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  const eT* mah_aux_mem = mah_aux.memptr();
  
  for(uword i=start_index; i <= end_index; ++i)
    {
    const eT* X_colptr = X.colptr(i);
    
    double best_dist = Datum<eT>::inf;
    uword  best_g    = 0;
    
    for(uword g=0; g < N_gaus; ++g)
      {
      const double dist = distance<eT,dist_id>::eval(N_dims, X_colptr, old_means.colptr(g), mah_aux_mem);
      
      // get_stream_err2() << "g: " << g << "   dist: " << dist << '\n';
      // old_means.col(g).print("old_means.col(g):");
      // vec tmp(old_means.colptr(g), old_means.n_rows);
      // tmp.print("tmp:");
      
      if(dist <= best_dist)  { best_dist = dist; best_g = g; }
      }
    
    // get_stream_err2() << "best_g: " << best_g << '\n';
    
    running_means[best_g]( X.unsafe_col(i), i );
    }
  }



//! multi-threaded implementation of Expectation-Maximisation, inspired by MapReduce
template<typename eT>
inline
bool
gmm_diag<eT>::em_iterate(const Mat<eT>& X, const uword max_iter, const eT var_floor, const bool verbose)
  {
  arma_extra_debug_sigprint();
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  if(verbose)
    {
    get_stream_err2().unsetf(ios::showbase);
    get_stream_err2().unsetf(ios::uppercase);
    get_stream_err2().unsetf(ios::showpos);
    get_stream_err2().unsetf(ios::scientific);
    
    get_stream_err2().setf(ios::right);
    get_stream_err2().setf(ios::fixed);
    }
  
  #if defined(_OPENMP)
    const arma_omp_state save_omp_state;
  #endif
  
  const umat boundaries = internal_gen_boundaries(X.n_cols);
  
  const uword n_threads = boundaries.n_cols;
  
  field< Mat<eT> > t_acc_means(n_threads); 
  field< Mat<eT> > t_acc_dcovs(n_threads);
  
  field< Col<eT> > t_acc_norm_lhoods(n_threads);
  field< Col<eT> > t_gaus_log_lhoods(n_threads);
  
  Col<eT>          t_progress_log_lhood(n_threads);
  
  for(uword t=0; t<n_threads; t++)
    {
    t_acc_means[t].set_size(N_dims, N_gaus);
    t_acc_dcovs[t].set_size(N_dims, N_gaus);
    
    t_acc_norm_lhoods[t].set_size(N_gaus);
    t_gaus_log_lhoods[t].set_size(N_gaus);
    }
  
  
  #if defined(_OPENMP)
    if(verbose)
      {
      get_stream_err2() << "gmm_diag::learn(): EM: n_threads: " << n_threads  << '\n';
      }
  #endif
  
  eT old_avg_log_p = -Datum<eT>::inf;
  
  for(uword iter=1; iter <= max_iter; ++iter)
    {
    init_constants();
    
    em_update_params(X, boundaries, t_acc_means, t_acc_dcovs, t_acc_norm_lhoods, t_gaus_log_lhoods, t_progress_log_lhood);
    
    em_fix_params(var_floor);
    
    const eT new_avg_log_p = mean(t_progress_log_lhood);
    
    if(verbose)
      {
      get_stream_err2() << "gmm_diag::learn(): EM: iteration: ";
      get_stream_err2().unsetf(ios::scientific);
      get_stream_err2().setf(ios::fixed);
      get_stream_err2().width(std::streamsize(4));
      get_stream_err2() << iter;
      get_stream_err2() << "   avg_log_p: ";
      get_stream_err2().unsetf(ios::fixed);
      //get_stream_err2().setf(ios::scientific);
      get_stream_err2() << new_avg_log_p << '\n';
      }
    
    if(is_finite(new_avg_log_p) == false)  { return false; }
    
    if(std::abs(old_avg_log_p - new_avg_log_p) <= Datum<eT>::eps)  { break; }
    
    
    old_avg_log_p = new_avg_log_p;
    }
  
  
  if(any(vectorise(dcovs) <= eT(0)))  { return false; }
  if(means.is_finite() == false    )  { return false; }
  if(dcovs.is_finite() == false    )  { return false; }
  if(hefts.is_finite() == false    )  { return false; }
  
  return true;
  }




template<typename eT>
inline
void
gmm_diag<eT>::em_update_params
  (
  const Mat<eT>&          X,
  const umat&             boundaries,
        field< Mat<eT> >& t_acc_means,
        field< Mat<eT> >& t_acc_dcovs,
        field< Col<eT> >& t_acc_norm_lhoods,
        field< Col<eT> >& t_gaus_log_lhoods,
        Col<eT>&          t_progress_log_lhood
  )
  {
  arma_extra_debug_sigprint();
  
  const uword n_threads = boundaries.n_cols;
  
  
  // em_generate_acc() is the "map" operation, which produces partial accumulators for means, diagonal covariances and hefts
    
  #if defined(_OPENMP)
    {
    #pragma omp parallel for
    for(uword t=0; t<n_threads; t++)
      {
      Mat<eT>& acc_means          = t_acc_means[t];
      Mat<eT>& acc_dcovs          = t_acc_dcovs[t];
      Col<eT>& acc_norm_lhoods    = t_acc_norm_lhoods[t];
      Col<eT>& gaus_log_lhoods    = t_gaus_log_lhoods[t];
      eT&      progress_log_lhood = t_progress_log_lhood[t];
      
      em_generate_acc(X, boundaries.at(0,t), boundaries.at(1,t), acc_means, acc_dcovs, acc_norm_lhoods, gaus_log_lhoods, progress_log_lhood);
      }
    }
  #else
    {
    em_generate_acc(X, boundaries.at(0,0), boundaries.at(1,0), t_acc_means[0], t_acc_dcovs[0], t_acc_norm_lhoods[0], t_gaus_log_lhoods[0], t_progress_log_lhood[0]);
    }
  #endif
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  Mat<eT>& final_acc_means = t_acc_means[0];
  Mat<eT>& final_acc_dcovs = t_acc_dcovs[0];
  
  Col<eT>& final_acc_norm_lhoods = t_acc_norm_lhoods[0];
  
  
  // the "reduce" operation, which combines the partial accumulators produced by the separate threads
  
  for(uword t=1; t<n_threads; t++)
    {
    final_acc_means += t_acc_means[t];
    final_acc_dcovs += t_acc_dcovs[t];
    
    final_acc_norm_lhoods += t_acc_norm_lhoods[t];
    }
  
  
  eT* hefts_mem = access::rw(hefts).memptr();
    
  for(uword g=0; g < N_gaus; ++g)
    {
    eT* mean_mem = access::rw(means).colptr(g);
    eT* dcov_mem = access::rw(dcovs).colptr(g);
    
    eT* acc_mean_mem = final_acc_means.colptr(g);
    eT* acc_dcov_mem = final_acc_dcovs.colptr(g);
    
    const eT acc_norm_lhood = final_acc_norm_lhoods[g];
    
    hefts_mem[g] = acc_norm_lhood / eT(X.n_cols);
    
    for(uword d=0; d < N_dims; ++d)
      {
      const eT tmp = acc_mean_mem[d] / acc_norm_lhood;
      
      mean_mem[d] = tmp;
      dcov_mem[d] = acc_dcov_mem[d] / acc_norm_lhood - tmp*tmp;
      }
    }
  }



template<typename eT>
inline
void
gmm_diag<eT>::em_generate_acc
  (
  const Mat<eT>& X,
  const uword    start_index,
  const uword      end_index,
        Mat<eT>& acc_means,
        Mat<eT>& acc_dcovs,
        Col<eT>& acc_norm_lhoods,
        Col<eT>& gaus_log_lhoods,
        eT&      progress_log_lhood
  )
  const
  {
  arma_extra_debug_sigprint();
  
  progress_log_lhood = eT(0);
  
  acc_means.zeros();
  acc_dcovs.zeros();
  
  acc_norm_lhoods.zeros();
  gaus_log_lhoods.zeros();
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  const eT* log_hefts_mem       = log_hefts.memptr();
        eT* gaus_log_lhoods_mem = gaus_log_lhoods.memptr();
  
  
  for(uword i=start_index; i <= end_index; i++)
    {
    const eT* x = X.colptr(i);
    
    for(uword g=0; g < N_gaus; ++g)
      {
      gaus_log_lhoods_mem[g] = internal_scalar_log_p(x, g) + log_hefts_mem[g];
      }
    
    eT log_lhood_sum = gaus_log_lhoods_mem[0];
    
    for(uword g=1; g < N_gaus; ++g)
      {
      log_lhood_sum = log_add_exp(log_lhood_sum, gaus_log_lhoods_mem[g]);
      }
    
    progress_log_lhood += log_lhood_sum;
    
    for(uword g=0; g < N_gaus; ++g)
      {
      const eT norm_lhood = std::exp(gaus_log_lhoods_mem[g] - log_lhood_sum);
      
      acc_norm_lhoods[g] += norm_lhood;
      
      eT* acc_mean_mem = acc_means.colptr(g);
      eT* acc_dcov_mem = acc_dcovs.colptr(g);
      
      for(uword d=0; d < N_dims; ++d)
        {
        const eT x_d = x[d];
        const eT y_d = x_d * norm_lhood;
        
        acc_mean_mem[d] += y_d;
        acc_dcov_mem[d] += y_d * x_d;  // equivalent to x_d * x_d * norm_lhood
        }
      }
    }
  
  progress_log_lhood /= eT((end_index - start_index) + 1);
  }



template<typename eT>
inline
void
gmm_diag<eT>::em_fix_params(const eT var_floor)
  {
  arma_extra_debug_sigprint();
  
  const uword N_dims = means.n_rows;
  const uword N_gaus = means.n_cols;
  
  for(uword g=0; g < N_gaus; ++g)
    {
    eT* dcov_mem = access::rw(dcovs).colptr(g);
    
    for(uword d=0; d < N_dims; ++d)
      {
      if(dcov_mem[d] < var_floor)  { dcov_mem[d] = var_floor; }
      }
    }
  
  const eT heft_sum = accu(hefts);
  
  if(heft_sum != eT(1))  { access::rw(hefts) / heft_sum; }
  }


}
