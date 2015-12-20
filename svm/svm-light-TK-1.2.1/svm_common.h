/************************************************************************/
/*                                                                      */
/*   svm_common.h                                                       */
/*                                                                      */
/*   Definitions and functions used in both svm_learn and svm_classify. */
/*                                                                      */
/*   Author: Thorsten Joachims                                          */
/*   Date: 02.07.02 							*/
/*                                                                      */
/*                                                                      */
/*   Modified by Alessandro Moschitti for the Tree Kernel Processing	*/
/*   Date: 25.10.04 							*/
/*                                                                      */
/*   Copyright (c) 2002  Thorsten Joachims - All rights reserved        */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

/* # define MICROSOFT */  /* uncomment, if compiling with Visual C++ */

#define MAXSHRINK 50000    /* maximum number of shrinking rounds */
#define MAX_NUMBER_OF_CHILDREN 100
#define MAX_PARSE_TREE_LENGTH 50000
#define MAX_NUMBER_OF_NODES 1000
#define MAX_NUMBER_OF_PAIRS 50000
#define MAX_PRODUCTION_LENGTH 1000
#define MAX_NUMBER_OF_TREES 20000

# include <stdio.h>
# include <ctype.h>
# include <math.h>
# include <string.h>
# include <stdlib.h>
# include <time.h> 
# include <float.h>

# define VERSION       "V5.01-TK-1.2" // TREE FOREST AND VECTOR SET
# define VERSION_DATE  "15.11.06"

# define CFLOAT  float       /* the type of float to use for caching */
                             /* kernel evaluations. Using float saves */
                             /* us some memory, but you can use double, too */
# define FNUM    long        /* the type used for storing feature ids */
# define FVAL    float       /* the type used for storing feature values */

# define LINEAR  0           /* linear kernel type */
# define POLY    1           /* polynoial kernel type */
# define RBF     2           /* rbf kernel type */
# define SIGMOID 3           /* sigmoid kernel type */

# define CLASSIFICATION 1    /* train classification model */
# define REGRESSION     2    /* train regression model */
# define RANKING        3    /* train ranking model */




typedef struct word {
  FNUM    wnum;	               /* word number */
  FVAL    weight;              /* word weight */
} WORD;

/* This Class Stores the Node information of the Syntactic Tree
*/


typedef struct TreeNodes{

   char *sName; // stores the name of the node
   struct TreeNodes **pChild; // stores the array of pointers to the childern of this node 
   int iNoOfChildren; // stores the number of childern of this node
   short nodeID;
   char *production;
   short pre_terminal;
		
} TreeNode;

typedef struct OrederedTreeNodes{

   char *sName; 	   // stores the production at the node
   struct TreeNodes *node; // stores the pointers to the corresponding node in the tree structure 
		
} OrderedTreeNode;

typedef struct forest{
          double  twonorm_PT;          /* The squared euclidian length of the tree*/
          TreeNode    *root;           /* The Parse-Tree */
          OrderedTreeNode 
              *orderedNodeSet;  /* Contain the ordered (by production) node list */
          short  listSize;		/* size of the above list */
} FOREST;

typedef struct setOfLinearVectors{
          double  twonorm_STD;          /* The squared vector norm*/
          double  twonorm_sq;           /* The squared euclidian length of vector*/
          WORD    *words;              /* The words/values in the document by  increasing word-number. */  
} VECTOR_SET;

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
  
  double twonorm_sq;
  
  VECTOR_SET **vectors;    /* A set of vectors */  

  FOREST **forest_vec; /* A set of trees */
  
  int num_of_trees; /* Number of trees in the forest*/
  int num_of_vectors; /* Number of trees in the forest*/

} DOC;

typedef struct nodepair {
  TreeNode    *Nx;	               
  TreeNode    *Nz;	               
} nodePair;

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
  long   svm_maxqpsize;        /* size q of working set */
  long   svm_newvarsinqp;      /* new variables to enter the working set 
				  in each iteration */
  double epsilon_crit;         /* tolerable error for distances used 
				  in stopping criterion */
  double epsilon_shrink;       /* how much a multiplier should be above 
				  zero for shrinking */
  long   svm_iter_to_shrink;   /* iterations h after which an example can
				  be removed by shrinking */
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
  double  tree_constant; /* Tree constant in t*K1 + K2 combined kernel*/
  double  lambda;    /* Decay factor in Tree kernel */
  char	  combination_type; /* '+' or '*' in Kernel combinations */
  long 	  second_kernel; /* second kernel in combination (K2) 0=linear, 1=poly, 2=rbf, 3=sigmoid, 4=custom*/
  long 	  first_kernel; /* second kernel in combination (K2) 0=linear, 1=poly, 2=rbf, 3=sigmoid, 4=custom*/
  char    vectorial_approach_standard_kernel;
  char    vectorial_approach_tree_kernel;
  float   sigma; 
  long    normalization; 
 
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
CFLOAT kernel(KERNEL_PARM *, DOC *, DOC *); 
double custom_kernel(KERNEL_PARM *, DOC *, DOC *); 
double sprod_ss(WORD *, WORD *);
WORD*  sub_ss(WORD *, WORD *); 
double model_length_s(MODEL *, KERNEL_PARM *);
void   clear_vector_n(double *, long);
void   add_vector_ns(double *, WORD *, double);
double sprod_ns(double *, WORD *);
void   add_weight_vector_to_linear_model(MODEL *);
void   read_model(char *, MODEL *, long, long);
void   read_documents(char *, DOC *, double *, long, long, long *, long *, KERNEL_PARM *);

int    parse_document(char *, DOC *, double *, long *, long, KERNEL_PARM *);
void   nol_ll(char *, long *, long *, long *);
long   minl(long, long);
long   maxl(long, long);
long   get_runtime(void);
void   *my_malloc(size_t); 
void   copyright_notice(void);

extern void evaluateNorma(KERNEL_PARM * kernel_parm, DOC *d);
extern double combination_kernel(KERNEL_PARM * kernel_parm, DOC * a, DOC * b);
extern double tree_kernel(KERNEL_PARM * kernel_parm, DOC * a, DOC * b, int i, int j);
extern double basic_kernel(KERNEL_PARM *kernel_parm, DOC *a, DOC *b, int i, int j);
extern double advanced_kernels(KERNEL_PARM * , DOC * , DOC * );
extern void extractOrderedListofTreeNodes(FOREST *);
extern void getStringTree(TreeNode *node, char *temp);
extern void writeTreeString(TreeNode *node);
extern void freeForest (DOC *);
extern double sprod(DOC *a, DOC *b);
extern double choose_tree_kernel(KERNEL_PARM *kernel_parm, DOC *a, DOC *b);

extern void read_tree_forest(char *line, DOC *example, int *pos);
extern void read_vector_set(char *line, DOC *example, long int max_words_doc, long int *totwords);
extern void freeForest(DOC *d);
extern void freeVectorSet(DOC *d);
extern void freeExample(DOC *d);

extern double sprod_i(DOC *a, DOC *b, int i, int j);
extern double SRL_re_ranking_CoNLL2006(KERNEL_PARM * kernel_parm, DOC * a, DOC * b);
extern double ACL2006_Entailment_kernel(KERNEL_PARM * kernel_parm, DOC * a, DOC * b);


# ifdef MICROSOFT
   int isnan(double);
# endif

extern long   verbosity;              /* verbosity level (0-4) */
extern long   kernel_cache_statistic;
extern double LAMBDA;
extern double SIGMA;

