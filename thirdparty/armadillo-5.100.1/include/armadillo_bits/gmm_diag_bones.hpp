// Copyright (C) 2014 Conrad Sanderson
// Copyright (C) 2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


struct gmm_dist_mode { const uword id;  inline explicit gmm_dist_mode(const uword in_id) : id(in_id) {} };

inline bool operator==(const gmm_dist_mode& a, const gmm_dist_mode& b) { return (a.id == b.id); }
inline bool operator!=(const gmm_dist_mode& a, const gmm_dist_mode& b) { return (a.id != b.id); }

struct gmm_dist_eucl : public gmm_dist_mode { inline gmm_dist_eucl() : gmm_dist_mode(1) {} };
struct gmm_dist_maha : public gmm_dist_mode { inline gmm_dist_maha() : gmm_dist_mode(2) {} };
struct gmm_dist_prob : public gmm_dist_mode { inline gmm_dist_prob() : gmm_dist_mode(3) {} };

static const gmm_dist_eucl eucl_dist;
static const gmm_dist_maha maha_dist;
static const gmm_dist_prob prob_dist;



struct gmm_seed_mode { const uword id; inline explicit gmm_seed_mode(const uword in_id) : id(in_id) {} };

inline bool operator==(const gmm_seed_mode& a, const gmm_seed_mode& b) { return (a.id == b.id); }
inline bool operator!=(const gmm_seed_mode& a, const gmm_seed_mode& b) { return (a.id != b.id); }

struct gmm_seed_keep_existing : public gmm_seed_mode { inline gmm_seed_keep_existing() : gmm_seed_mode(1) {} };
struct gmm_seed_static_subset : public gmm_seed_mode { inline gmm_seed_static_subset() : gmm_seed_mode(2) {} };
struct gmm_seed_static_spread : public gmm_seed_mode { inline gmm_seed_static_spread() : gmm_seed_mode(3) {} };
struct gmm_seed_random_subset : public gmm_seed_mode { inline gmm_seed_random_subset() : gmm_seed_mode(4) {} };
struct gmm_seed_random_spread : public gmm_seed_mode { inline gmm_seed_random_spread() : gmm_seed_mode(5) {} };

static const gmm_seed_keep_existing keep_existing;
static const gmm_seed_static_subset static_subset;
static const gmm_seed_static_spread static_spread;
static const gmm_seed_random_subset random_subset;
static const gmm_seed_random_spread random_spread;


namespace gmm_priv
{

struct gmm_empty_arg {};


#if defined(_OPENMP)
  struct arma_omp_state
    {
    const int orig_dynamic_state;
    
    inline  arma_omp_state() : orig_dynamic_state(omp_get_dynamic()) { omp_set_dynamic(0); }
    inline ~arma_omp_state() { omp_set_dynamic(orig_dynamic_state); }
    };
#else
  struct arma_omp_state {};
#endif



template<typename eT>
class gmm_diag
  {
  public:
  
  arma_aligned const Mat<eT> means;
  arma_aligned const Mat<eT> dcovs;
  arma_aligned const Row<eT> hefts;
  
  //
  //
  
  inline ~gmm_diag();
  inline  gmm_diag();
  
  inline                  gmm_diag(const gmm_diag& x);
  inline const gmm_diag& operator=(const gmm_diag& x);
  
  inline      gmm_diag(const uword in_n_dims, const uword in_n_gaus);
  inline void    reset(const uword in_n_dims, const uword in_n_gaus);
  inline void    reset();
  
  template<typename T1, typename T2, typename T3>
  inline void set_params(const Base<eT,T1>& in_means, const Base<eT,T2>& in_dcovs, const Base<eT,T3>& in_hefts);
  
  template<typename T1> inline void set_means(const Base<eT,T1>& in_means);
  template<typename T1> inline void set_dcovs(const Base<eT,T1>& in_dcovs);
  template<typename T1> inline void set_hefts(const Base<eT,T1>& in_hefts);
  
  inline uword n_dims() const;
  inline uword n_gaus() const;
  
  inline bool load(const std::string name);
  inline bool save(const std::string name) const;
  
  inline Col<eT> generate()              const;
  inline Mat<eT> generate(const uword N) const;
  
  template<typename T1> inline eT      log_p(const T1& expr, const gmm_empty_arg& junk1 = gmm_empty_arg(), typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == true ))>::result* junk2 = 0) const;
  template<typename T1> inline eT      log_p(const T1& expr, const uword gaus_id,                          typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == true ))>::result* junk2 = 0) const;
  
  template<typename T1> inline Row<eT> log_p(const T1& expr, const gmm_empty_arg& junk1 = gmm_empty_arg(), typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == false))>::result* junk2 = 0) const;
  template<typename T1> inline Row<eT> log_p(const T1& expr, const uword gaus_id,                          typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == false))>::result* junk2 = 0) const;
  
  template<typename T1> inline eT  avg_log_p(const Base<eT,T1>& expr)                      const;
  template<typename T1> inline eT  avg_log_p(const Base<eT,T1>& expr, const uword gaus_id) const;
  
  template<typename T1> inline uword   assign(const T1& expr, const gmm_dist_mode& dist, typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == true ))>::result* junk = 0) const;
  template<typename T1> inline urowvec assign(const T1& expr, const gmm_dist_mode& dist, typename enable_if<((is_arma_type<T1>::value) && (resolves_to_colvector<T1>::value == false))>::result* junk = 0) const;
  
  template<typename T1> inline urowvec  raw_hist(const Base<eT,T1>& expr, const gmm_dist_mode& dist_mode) const;
  template<typename T1> inline Row<eT> norm_hist(const Base<eT,T1>& expr, const gmm_dist_mode& dist_mode) const;
  
  template<typename T1>
  inline
  bool
  learn
    (
    const Base<eT,T1>&    data,
    const uword           n_gaus,
    const gmm_dist_mode&  dist_mode,
    const gmm_seed_mode&  seed_mode,
    const uword           km_iter,
    const uword           em_iter,
    const eT              var_floor,
    const bool            print_mode
    );
  
  
  //
  
  protected:
  
  
  arma_aligned Row<eT> log_det_etc;
  arma_aligned Row<eT> log_hefts;
  arma_aligned Col<eT> mah_aux;
  
  //
  
  inline void init(const gmm_diag& x);
  
  inline void init(const uword in_n_dim, const uword in_n_gaus);
  
  inline void init_constants();

  inline umat internal_gen_boundaries(const uword N) const;

  inline eT internal_scalar_log_p(const eT* x                     ) const;
  inline eT internal_scalar_log_p(const eT* x, const uword gaus_id) const;
  
  template<typename T1> inline Row<eT> internal_vec_log_p(const T1& X                     ) const;
  template<typename T1> inline Row<eT> internal_vec_log_p(const T1& X, const uword gaus_id) const;
  
  template<typename T1> inline eT internal_avg_log_p(const T1& X                     ) const;
  template<typename T1> inline eT internal_avg_log_p(const T1& X, const uword gaus_id) const;
  
  template<typename T1> inline uword internal_scalar_assign(const T1& X, const gmm_dist_mode& dist_mode) const;
  
  template<typename T1> inline void internal_vec_assign(urowvec& out, const T1& X, const gmm_dist_mode& dist_mode) const;
  
  inline void internal_raw_hist(urowvec& hist, const Mat<eT>& X, const gmm_dist_mode& dist_mode) const;
  
  //
  
  template<uword dist_id> inline void generate_initial_means(const Mat<eT>& X, const gmm_seed_mode& seed);
  
  template<uword dist_id> inline void generate_initial_dcovs_and_hefts(const Mat<eT>& X, const eT var_floor);
  
  template<uword dist_id> inline bool km_iterate(const Mat<eT>& X, const uword max_iter, const bool verbose);
  
  template<uword dist_id> inline void km_update_stats(const Mat<eT>& X, const uword start_index, const uword end_index, const Mat<eT>& old_means, field< running_mean_vec<eT> >& running_means) const;
  
  //
  
  inline bool em_iterate(const Mat<eT>& X, const uword max_iter, const eT var_floor, const bool verbose);
  
  inline void em_update_params(const Mat<eT>& X, const umat& boundaries, field< Mat<eT> >& t_acc_means, field< Mat<eT> >& t_acc_dcovs, field< Col<eT> >& t_acc_norm_lhoods, field< Col<eT> >& t_gaus_log_lhoods, Col<eT>& t_progress_log_lhoods);
  
  inline void em_generate_acc(const Mat<eT>& X, const uword start_index, const uword end_index, Mat<eT>& acc_means, Mat<eT>& acc_dcovs, Col<eT>& acc_norm_lhoods, Col<eT>& gaus_log_lhoods, eT& progress_log_lhood) const;
  
  inline void em_fix_params(const eT var_floor);
  };

}


typedef gmm_priv::gmm_diag<double>  gmm_diag;
typedef gmm_priv::gmm_diag<float>  fgmm_diag;

