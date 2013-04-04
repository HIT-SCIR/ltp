/************************************************************************/
/*                                                                      */
/*   svm_common.h                                                       */
/*                                                                      */
/*   Definitions and functions used in both svm_learn and svm_classify. */
/*                                                                      */
/*   Author: Thorsten Joachims                                          */
/*   Date: 02.07.02                                                     */
/*                                                                      */
/*   Copyright (c) 2002  Thorsten Joachims - All rights reserved        */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#ifndef SVM_COMMON
#define SVM_COMMON

# include <stdio.h>
# include <ctype.h>
# include <math.h>
# include <string.h>
# include <stdlib.h>
# include <time.h> 
# include <float.h>

# define VERSION       "V6.02"
# define VERSION_DATE  "14.08.08"

# define CFLOAT  float       /* the type of float to use for caching */
                             /* kernel evaluations. Using float saves */
                             /* us some memory, but you can use double, too */
# define FNUM    long        /* the type used for storing feature ids */
# define FVAL    float       /* the type used for storing feature values */
# define MAXFEATNUM 99999999 /* maximum feature number (must be in
			  	valid range of FNUM type and long int!) */

# define LINEAR  0           /* linear kernel type */
# define POLY    1           /* polynoial kernel type */
# define RBF     2           /* rbf kernel type */
# define SIGMOID 3           /* sigmoid kernel type */

# define CLASSIFICATION 1    /* train classification model */
# define REGRESSION     2    /* train regression model */
# define RANKING        3    /* train ranking model */
# define OPTIMIZATION   4    /* train on general set of constraints */

# define MAXSHRINK     50000    /* maximum number of shrinking rounds */

typedef struct word {
  FNUM    wnum;	               /* word number */
  FVAL    weight;              /* word weight */
} WORD;

typedef struct svector {
  WORD    *words;              /* The features/values in the vector by
				  increasing feature-number. Feature
				  numbers that are skipped are
				  interpreted as having value zero. */
  double  twonorm_sq;          /* The squared euclidian length of the
                                  vector. Used to speed up the RBF kernel. */
  char    *userdefined;        /* You can put additional information
				  here. This can be useful, if you are
				  implementing your own kernel that
				  does not work with feature/values
				  representations (for example a
				  string kernel). By default,
				  svm-light will put here the string
				  after the # sign from each line of
				  the input file. */
  long    kernel_id;           /* Feature vectors with different
				  kernel_id's are orthogonal (ie. the
				  feature number do not match). This
				  is used for computing component
				  kernels for linear constraints which
				  are a sum of several different
				  weight vectors. (currently not
				  implemented). */
  struct svector *next;        /* Let's you set up a list of SVECTOR's
				  for linear constraints which are a
				  sum of multiple feature
				  vectors. List is terminated by
				  NULL. */
  double  factor;              /* Factor by which this feature vector
				  is multiplied in the sum. */
} SVECTOR;

typedef struct doc {
  long    docnum;              /* Document ID. This has to be the position of 
                                  the document in the training set array. */
  long    queryid;             /* for learning rankings, constraints are 
				  generated for documents with the same 
				  queryID. */
  double  costfactor;          /* Scales the cost of misclassifying this
				  document by this factor. The effect of this
				  value is, that the upper bound on the alpha
				  for this example is scaled by this factor.
				  The factors are set by the feature 
				  'cost:<val>' in the training data. */
  long    slackid;             /* Index of the slack variable
				  corresponding to this
				  constraint. All constraints with the
				  same slackid share the same slack
				  variable. This can only be used for
				  svm_learn_optimization. */
  SVECTOR *fvec;               /* Feature vector of the example. The
				  feature vector can actually be a
				  list of feature vectors. For
				  example, the list will have two
				  elements, if this DOC is a
				  preference constraint. The one
				  vector that is supposed to be ranked
				  higher, will have a factor of +1,
				  the lower ranked one should have a
				  factor of -1. */
} DOC;

typedef struct learn_parm {
  long   type;                 /* selects between regression and
				  classification */
  double svm_c;                /* upper bound C on alphas */
  double eps;                  /* regression epsilon (eps=1.0 for
				  classification */
  double svm_costratio;        /* factor to multiply C for positive examples */
  double transduction_posratio;/* fraction of unlabeled examples to be */
                               /* classified as positives */
  long   biased_hyperplane;    /* if nonzero, use hyperplane w*x+b=0 
				  otherwise w*x=0 */
  long   sharedslack;          /* if nonzero, it will use the shared
                                  slack variable mode in
                                  svm_learn_optimization. It requires
                                  that the slackid is set for every
                                  training example */
  long   svm_maxqpsize;        /* size q of working set */
  long   svm_newvarsinqp;      /* new variables to enter the working set 
				  in each iteration */
  long   kernel_cache_size;    /* size of kernel cache in megabytes */
  double epsilon_crit;         /* tolerable error for distances used 
				  in stopping criterion */
  double epsilon_shrink;       /* how much a multiplier should be above 
				  zero for shrinking */
  long   svm_iter_to_shrink;   /* iterations h after which an example can
				  be removed by shrinking */
  long   maxiter;              /* number of iterations after which the
				  optimizer terminates, if there was
				  no progress in maxdiff */
  long   remove_inconsistent;  /* exclude examples with alpha at C and 
				  retrain */
  long   skip_final_opt_check; /* do not check KT-Conditions at the end of
				  optimization for examples removed by 
				  shrinking. WARNING: This might lead to 
				  sub-optimal solutions! */
  long   compute_loo;          /* if nonzero, computes leave-one-out
				  estimates */
  double rho;                  /* parameter in xi/alpha-estimates and for
				  pruning leave-one-out range [1..2] */
  long   xa_depth;             /* parameter in xi/alpha-estimates upper
				  bounding the number of SV the current
				  alpha_t is distributed over */
  char predfile[200];          /* file for predicitions on unlabeled examples
				  in transduction */
  char alphafile[200];         /* file to store optimal alphas in. use  
				  empty string if alphas should not be 
				  output */

  /* you probably do not want to touch the following */
  double epsilon_const;        /* tolerable error on eq-constraint */
  double epsilon_a;            /* tolerable error on alphas at bounds */
  double opt_precision;        /* precision of solver, set to e.g. 1e-21 
				  if you get convergence problems */

  /* the following are only for internal use */
  long   svm_c_steps;          /* do so many steps for finding optimal C */
  double svm_c_factor;         /* increase C by this factor every step */
  double svm_costratio_unlab;
  double svm_unlabbound;
  double *svm_cost;            /* individual upper bounds for each var */
  long   totwords;             /* number of features */
} LEARN_PARM;

typedef struct kernel_parm {
  long    kernel_type;   /* 0=linear, 1=poly, 2=rbf, 3=sigmoid, 4=custom */
  long    poly_degree;
  double  rbf_gamma;
  double  coef_lin;
  double  coef_const;
  char    custom[50];    /* for user supplied kernel */
} KERNEL_PARM;

typedef struct model {
  long    sv_num;	
  long    at_upper_bound;
  double  b;
  DOC     **supvec;
  double  *alpha;
  long    *index;       /* index from docnum to position in model */
  long    totwords;     /* number of features */
  long    totdoc;       /* number of training documents */
  KERNEL_PARM kernel_parm; /* kernel */

  /* the following values are not written to file */
  double  loo_error,loo_recall,loo_precision; /* leave-one-out estimates */
  double  xa_error,xa_recall,xa_precision;    /* xi/alpha estimates */
  double  *lin_weights;                       /* weights for linear case using
						 folding */
  double  maxdiff;                            /* precision, up to which this 
						 model is accurate */
} MODEL;

typedef struct quadratic_program {
  long   opt_n;            /* number of variables */
  long   opt_m;            /* number of linear equality constraints */
  double *opt_ce,*opt_ce0; /* linear equality constraints */
  double *opt_g;           /* hessian of objective */
  double *opt_g0;          /* linear part of objective */
  double *opt_xinit;       /* initial value for variables */
  double *opt_low,*opt_up; /* box constraints */
} QP;

typedef struct kernel_cache {
  long   *index;  /* cache some kernel evalutations */
  CFLOAT *buffer; /* to improve speed */
  long   *invindex;
  long   *active2totdoc;
  long   *totdoc2active;
  long   *lru;
  long   *occu;
  long   elems;
  long   max_elems;
  long   time;
  long   activenum;
  long   buffsize;
} KERNEL_CACHE;


typedef struct timing_profile {
  long   time_kernel;
  long   time_opti;
  long   time_shrink;
  long   time_update;
  long   time_model;
  long   time_check;
  long   time_select;
} TIMING;

typedef struct shrink_state {
  long   *active;
  long   *inactive_since;
  long   deactnum;
  double **a_history;  /* for shrinking with non-linear kernel */
  long   maxhistory;
  double *last_a;      /* for shrinking with linear kernel */
  double *last_lin;    /* for shrinking with linear kernel */
} SHRINK_STATE;

double classify_example(MODEL *, DOC *);
double classify_example_linear(MODEL *, DOC *);
double kernel(KERNEL_PARM *, DOC *, DOC *); 
double single_kernel(KERNEL_PARM *, SVECTOR *, SVECTOR *); 
double custom_kernel(KERNEL_PARM *, SVECTOR *, SVECTOR *); 
SVECTOR *create_svector(WORD *, char *, double);
SVECTOR *copy_svector(SVECTOR *);
void   free_svector(SVECTOR *);
double    sprod_ss(SVECTOR *, SVECTOR *);
SVECTOR*  sub_ss(SVECTOR *, SVECTOR *); 
SVECTOR*  add_ss(SVECTOR *, SVECTOR *); 
SVECTOR*  add_list_ss(SVECTOR *); 
void      append_svector_list(SVECTOR *a, SVECTOR *b);
SVECTOR*  smult_s(SVECTOR *, double);
int       featvec_eq(SVECTOR *, SVECTOR *); 
double model_length_s(MODEL *, KERNEL_PARM *);
void   clear_vector_n(double *, long);
void   add_vector_ns(double *, SVECTOR *, double);
double sprod_ns(double *, SVECTOR *);
void   add_weight_vector_to_linear_model(MODEL *);
DOC    *create_example(long, long, long, double, SVECTOR *);
void   free_example(DOC *, long);
MODEL  *read_model(char *);
MODEL  *copy_model(MODEL *);
void   free_model(MODEL *, int);
void   read_documents(char *, DOC ***, double **, long *, long *);
int    parse_document(char *, WORD *, double *, long *, long *, double *, long *, long, char **);
double *read_alphas(char *,long);
void   nol_ll(char *, long *, long *, long *);
long   minl(long, long);
long   maxl(long, long);
long   get_runtime(void);
int    space_or_null(int);
void   *my_malloc(size_t); 
void   copyright_notice(void);
# ifdef _MSC_VER
   int isnan(double);
# endif

extern long   verbosity;              /* verbosity level (0-4) */
extern long   kernel_cache_statistic;

#endif
