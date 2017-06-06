/***********************************************************************/
/*                                                                     */
/*   svm_learn.h                                                       */
/*                                                                     */
/*   Declarations for learning module of Support Vector Machine.       */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 02.07.02                                                    */
/*                                                                     */
/*   Copyright (c) 2002  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

void   svm_learn_classification(DOC *, double *, long, long, LEARN_PARM *, 
				KERNEL_PARM *, KERNEL_CACHE *, MODEL *);
void   svm_learn_regression(DOC *, double *, long, long, LEARN_PARM *, 
			    KERNEL_PARM *, KERNEL_CACHE *, MODEL *);
void   svm_learn_ranking(DOC *, double *, long, long, LEARN_PARM *, 
			 KERNEL_PARM *, KERNEL_CACHE *, MODEL *);
long   optimize_to_convergence(DOC *, long *, long, long, LEARN_PARM *,
			       KERNEL_PARM *, KERNEL_CACHE *, SHRINK_STATE *,
			       MODEL *, long *, long *, double *,
			       double *, double *,
			       TIMING *, double *, long, long);
double compute_objective_function(double *, double *, double *, double,
				  long *, long *);
void   clear_index(long *);
void   add_to_index(long *, long);
long   compute_index(long *,long, long *);
void   optimize_svm(DOC *, long *, long *, long *, long *, MODEL *, long, 
		    long *, long, double *, double *, double *, LEARN_PARM *, CFLOAT *, 
		    KERNEL_PARM *, QP *, double *);
void   compute_matrices_for_optimization(DOC *, long *, long *, long *, 
					 long *, long *, MODEL *, double *, 
					 double *, double *, long, long, LEARN_PARM *, 
					 CFLOAT *, KERNEL_PARM *, QP *);
long   calculate_svm_model(DOC *, long *, long *, double *, double *, 
			   double *, double *, LEARN_PARM *, long *,
			   long *, MODEL *);
long   check_optimality(MODEL *, long *, long *, double *, double *,
			double *, long, 
			LEARN_PARM *,double *, double, long *, long *, long *,
			long *, long, KERNEL_PARM *);
long   identify_inconsistent(double *, long *, long *, long, LEARN_PARM *, 
			     long *, long *);
long   identify_misclassified(double *, long *, long *, long,
			      MODEL *, long *, long *);
long   identify_one_misclassified(double *, long *, long *, long,
				  MODEL *, long *, long *);
long   incorporate_unlabeled_examples(MODEL *, long *,long *, long *,
				      double *, double *, long, double *,
				      long *, long *, long, KERNEL_PARM *,
				      LEARN_PARM *);
void   update_linear_component(DOC *, long *, long *, double *, double *, 
			       long *, long, long, KERNEL_PARM *, 
			       KERNEL_CACHE *, double *,
			       CFLOAT *, double *);
long   select_next_qp_subproblem_grad(long *, long *, double *, 
				      double *, double *, long,
				      long, LEARN_PARM *, long *, long *, 
				      long *, double *, long *, KERNEL_CACHE *,
				      long *, long *);
long   select_next_qp_subproblem_grad_cache(long *, long *, double *, double *,
					    double *, long, long, LEARN_PARM *, long *, 
					    long *, long *, double *, long *,
					    KERNEL_CACHE *, long *, long *);
long   select_next_qp_subproblem_rand(long *, long *, double *, 
				      double *, double *, long,
				      long, LEARN_PARM *, long *, long *, 
				      long *, double *, long *, KERNEL_CACHE *,
				      long *, long *, long);
void   select_top_n(double *, long, long *, long);
void   init_shrink_state(SHRINK_STATE *, long, long);
void   shrink_state_cleanup(SHRINK_STATE *);
long   shrink_problem(LEARN_PARM *, SHRINK_STATE *, KERNEL_PARM *, long *, 
		      long *, long, long, long, double *, long *);
void   reactivate_inactive_examples(long *, long *, double *, SHRINK_STATE *,
				    double *, double*, long, long, long, LEARN_PARM *, 
				    long *, DOC *, KERNEL_PARM *,
				    KERNEL_CACHE *, MODEL *, CFLOAT *, 
				    double *, double *);

/* cache kernel evalutations to improve speed */
void   get_kernel_row(KERNEL_CACHE *,DOC *, long, long, long *, CFLOAT *, 
		      KERNEL_PARM *);
void   cache_kernel_row(KERNEL_CACHE *,DOC *, long, KERNEL_PARM *);
void   cache_multiple_kernel_rows(KERNEL_CACHE *,DOC *, long *, long, 
				  KERNEL_PARM *);
void   kernel_cache_shrink(KERNEL_CACHE *,long, long, long *);
void   kernel_cache_init(KERNEL_CACHE *,long, long);
void   kernel_cache_reset_lru(KERNEL_CACHE *);
void   kernel_cache_cleanup(KERNEL_CACHE *);
long   kernel_cache_malloc(KERNEL_CACHE *);
void   kernel_cache_free(KERNEL_CACHE *,long);
long   kernel_cache_free_lru(KERNEL_CACHE *);
CFLOAT *kernel_cache_clean_and_malloc(KERNEL_CACHE *,long);
long   kernel_cache_touch(KERNEL_CACHE *,long);
long   kernel_cache_check(KERNEL_CACHE *,long);

void compute_xa_estimates(MODEL *, long *, long *, long, DOC *, 
			  double *, double *, KERNEL_PARM *, 
			  LEARN_PARM *, double *, double *, double *);
double xa_estimate_error(MODEL *, long *, long *, long, DOC *, 
			 double *, double *, KERNEL_PARM *, 
			 LEARN_PARM *);
double xa_estimate_recall(MODEL *, long *, long *, long, DOC *, 
			  double *, double *, KERNEL_PARM *, 
			  LEARN_PARM *);
double xa_estimate_precision(MODEL *, long *, long *, long, DOC *, 
			     double *, double *, KERNEL_PARM *, 
			     LEARN_PARM *);
void avg_similarity_of_sv_of_one_class(MODEL *, DOC *, double *, long *, KERNEL_PARM *, double *, double *);
double most_similar_sv_of_same_class(MODEL *, DOC *, double *, long, long *, KERNEL_PARM *, LEARN_PARM *);
double distribute_alpha_t_greedily(long *, long, DOC *, double *, long, long *, KERNEL_PARM *, LEARN_PARM *, double);
double distribute_alpha_t_greedily_noindex(MODEL *, DOC *, double *, long, long *, KERNEL_PARM *, LEARN_PARM *, double); 
void estimate_transduction_quality(MODEL *, long *, long *, long, DOC *, double *);
double estimate_margin_vcdim(MODEL *, double, double, KERNEL_PARM *);
double estimate_sphere(MODEL *, KERNEL_PARM *);
double estimate_r_delta_average(DOC *, long, KERNEL_PARM *); 
double estimate_r_delta(DOC *, long, KERNEL_PARM *); 
double length_of_longest_document_vector(DOC *, long, KERNEL_PARM *); 

void   write_model(char *, MODEL *);
void   write_prediction(char *, MODEL *, double *, double *, long *, long *,
			long, LEARN_PARM *);
void   write_alphas(char *, double *, long *, long);

typedef struct cache_parm_s {
  KERNEL_CACHE *kernel_cache;
  CFLOAT *cache;
  DOC *docs; 
  long m;
  KERNEL_PARM *kernel_parm;
  long offset,stepsize;
} cache_parm_t;

