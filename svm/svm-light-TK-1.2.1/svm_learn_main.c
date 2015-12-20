/***********************************************************************/
/*                                                                     */
/*   svm_learn_main.c                                                  */
/*                                                                     */
/*   Command line interface to the learning module of the              */
/*   Support Vector Machine.                                           */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 02.07.02                                                    */
/*                                                                     */
/*   Modified by Alessandro Moschitti                                  */
/*   Date: 15.11.06                                                    */
/*                                                                     */
/*   Copyright (c) 2000  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/


/* uncomment, if you want to use svm-learn out of C++ */
/* extern "C" { */
# include "svm_common.h"
# include "svm_learn.h"
/* } */

char docfile[200];           /* file with training examples */
char modelfile[200];         /* file for resulting classifier */

void   read_input_parameters(int, char **, char *, char *,long *, long *, 
			     LEARN_PARM *, KERNEL_PARM *);
void   wait_any_key();
void   print_help();

int main (int argc, char* argv[])
{  
  DOC *docs;  /* training examples */
  long max_docs,max_words_doc;
  long totwords,totdoc,ll,i;
  long kernel_cache_size;
  double *target;
  KERNEL_CACHE kernel_cache;
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  MODEL model;

  read_input_parameters(argc,argv,docfile,modelfile,&verbosity,
			&kernel_cache_size,&learn_parm,&kernel_parm);

  /* STANDARD SVM KERNELS */
      
  LAMBDA = kernel_parm.lambda; // to make faster the kernel evaluation
  SIGMA = kernel_parm.sigma;

  if(verbosity>=1) {
    printf("Scanning examples..."); fflush(stdout);
  }
  nol_ll(docfile,&max_docs,&max_words_doc,&ll); /* scan size of input file */
  max_words_doc+=10;
  ll+=10;
  max_docs+=2;
  if(verbosity>=1) {
    printf("done\n"); fflush(stdout);
  }

  docs = (DOC *)my_malloc(sizeof(DOC)*max_docs);         /* feature vectors */
  target = (double *)my_malloc(sizeof(double)*max_docs); /* target values */
 //printf("\nMax docs: %ld, approximated number of feature occurences %ld, maximal length of a line %ld\n\n",max_docs,max_words_doc,ll);
  read_documents(docfile,docs,target,max_words_doc,ll,&totwords,&totdoc,&kernel_parm);
  printf("\nNumber of examples: %ld, linear space size: %ld\n\n",totdoc,totwords);
 
 //if(kernel_parm.kernel_type==5) totwords=totdoc; // The number of features is proportional to the number of parse-trees, i.e. totdoc 
  				                 // or should we still use totwords to approximate svm_maxqpsize for the Tree Kernel (see hideo.c) ???????

  if(kernel_parm.kernel_type == LINEAR) { /* don't need the cache */
    if(learn_parm.type == CLASSIFICATION) {
      svm_learn_classification(docs,target,totdoc,totwords,&learn_parm,
			       &kernel_parm,NULL,&model);
    }
    else if(learn_parm.type == REGRESSION) {
      svm_learn_regression(docs,target,totdoc,totwords,&learn_parm,
			   &kernel_parm,NULL,&model);
    }
    else if(learn_parm.type == RANKING) {
      svm_learn_ranking(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,NULL,&model);
    }
  }
  else {
    if(learn_parm.type == CLASSIFICATION) {
      /* Always get a new kernel cache. It is not possible to use the
         same cache for two different training runs */
      kernel_cache_init(&kernel_cache,totdoc,kernel_cache_size);
      svm_learn_classification(docs,target,totdoc,totwords,&learn_parm,
			       &kernel_parm,&kernel_cache,&model);
      /* Free the memory used for the cache. */
      kernel_cache_cleanup(&kernel_cache);
    }
    else if(learn_parm.type == REGRESSION) {
      /* Always get a new kernel cache. It is not possible to use the
         same cache for two different training runs */
      kernel_cache_init(&kernel_cache,2*totdoc,kernel_cache_size);
      svm_learn_regression(docs,target,totdoc,totwords,&learn_parm,
			   &kernel_parm,&kernel_cache,&model);
      /* Free the memory used for the cache. */
      kernel_cache_cleanup(&kernel_cache);
    }
    else if(learn_parm.type == RANKING) {
      printf("Learning rankings is not implemented for non-linear kernels in this version!\n");
      exit(1);
    }
  }

  /* Warning: The model contains references to the original data 'docs'.
     If you want to free the original data, and only keep the model, you 
     have to make a deep copy of 'model'. */
  write_model(modelfile,&model);

  free(model.supvec);
  free(model.alpha);
  free(model.index);
  
   for(i=0;i<totdoc;i++){
     freeExample(&docs[i]);
   }
  
  free(docs);
  free(target);

  return(0);
}

/*---------------------------------------------------------------------------*/

void read_input_parameters(int argc,char *argv[],char *docfile,char *modelfile,
			   long *verbosity,long *kernel_cache_size,
			   LEARN_PARM *learn_parm,KERNEL_PARM *kernel_parm)
{
  long i;
  char type[100];
  
  /* set default */
  strcpy (modelfile, "svm_model");
  strcpy (learn_parm->predfile, "trans_predictions");
  strcpy (learn_parm->alphafile, "");
  (*verbosity)=1;
  learn_parm->biased_hyperplane=1;
  learn_parm->remove_inconsistent=0;
  learn_parm->skip_final_opt_check=0;
  learn_parm->svm_maxqpsize=10;
  learn_parm->svm_newvarsinqp=0;
  learn_parm->svm_iter_to_shrink=-9999;
  (*kernel_cache_size)=40;
  learn_parm->svm_c=0.0;
  learn_parm->eps=0.1;
  learn_parm->transduction_posratio=-1.0;
  learn_parm->svm_costratio=1.0;
  learn_parm->svm_costratio_unlab=1.0;
  learn_parm->svm_unlabbound=1E-5;
  learn_parm->epsilon_crit=0.001;
  learn_parm->epsilon_a=1E-15;
  learn_parm->compute_loo=0;
  learn_parm->rho=1.0;
  learn_parm->xa_depth=0;
  kernel_parm->kernel_type=0;
  kernel_parm->poly_degree=3;
  kernel_parm->rbf_gamma=1.0;
  kernel_parm->coef_lin=1;
  kernel_parm->coef_const=1;
  kernel_parm->lambda=.4;
  kernel_parm->tree_constant=1;
  kernel_parm->second_kernel=1;
  kernel_parm->first_kernel=1; 
  kernel_parm->normalization=3;
  kernel_parm->combination_type='T'; //no combination
  kernel_parm->vectorial_approach_standard_kernel='S';
  kernel_parm->vectorial_approach_tree_kernel='S';
  kernel_parm->sigma=1; // Default Duffy and Collins Kernel 
 
  strcpy(kernel_parm->custom,"empty");
  strcpy(type,"c");

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) 
      { 
      case '?': print_help(); exit(0);
      case 'z': i++; strcpy(type,argv[i]); break;
      case 'v': i++; (*verbosity)=atol(argv[i]); break;
      case 'b': i++; learn_parm->biased_hyperplane=atol(argv[i]); break;
      case 'i': i++; learn_parm->remove_inconsistent=atol(argv[i]); break;
      case 'f': i++; learn_parm->skip_final_opt_check=!atol(argv[i]); break;
      case 'q': i++; learn_parm->svm_maxqpsize=atol(argv[i]); break;
      case 'n': i++; learn_parm->svm_newvarsinqp=atol(argv[i]); break;
      case 'h': i++; learn_parm->svm_iter_to_shrink=atol(argv[i]); break;
      case 'm': i++; (*kernel_cache_size)=atol(argv[i]); break;
      case 'c': i++; learn_parm->svm_c=atof(argv[i]); break;
      case 'w': i++; learn_parm->eps=atof(argv[i]); break;
      case 'p': i++; learn_parm->transduction_posratio=atof(argv[i]); break;
      case 'j': i++; learn_parm->svm_costratio=atof(argv[i]); break;
      case 'e': i++; learn_parm->epsilon_crit=atof(argv[i]); break;
      case 'o': i++; learn_parm->rho=atof(argv[i]); break;
      case 'k': i++; learn_parm->xa_depth=atol(argv[i]); break;
      case 'x': i++; learn_parm->compute_loo=atol(argv[i]); break;
      case 't': i++; kernel_parm->kernel_type=atol(argv[i]); break;
      case 'd': i++; kernel_parm->poly_degree=atol(argv[i]); break;
      case 'g': i++; kernel_parm->rbf_gamma=atof(argv[i]); break;
      case 's': i++; kernel_parm->coef_lin=atof(argv[i]); break;
      case 'r': i++; kernel_parm->coef_const=atof(argv[i]); break;
      case 'u': i++; strcpy(kernel_parm->custom,argv[i]); break;
      case 'l': i++; strcpy(learn_parm->predfile,argv[i]); break;
      case 'a': i++; strcpy(learn_parm->alphafile,argv[i]); break;
      case 'L': i++; kernel_parm->lambda=atof(argv[i]); break;
      case 'T': i++; kernel_parm->tree_constant=atof(argv[i]); break;
      case 'C': i++; kernel_parm->combination_type=*argv[i]; break;
      case 'F': i++; kernel_parm->first_kernel=atoi(argv[i]); break;
      case 'S': i++; kernel_parm->second_kernel=atoi(argv[i]); break;
      case 'V': i++; kernel_parm->vectorial_approach_standard_kernel=*argv[i]; break;
      case 'W': i++; kernel_parm->vectorial_approach_tree_kernel=*argv[i]; break;
      case 'D': i++; kernel_parm->sigma=atof(argv[i]); break; 
      case 'N': i++; kernel_parm->normalization=atoi(argv[i]); break; 


      default: printf("\nUnrecognized option %s!\n\n",argv[i]);
	       print_help();
	       exit(0);
      }
  }
  if(i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  

  strcpy (docfile, argv[i]);
  if((i+1)<argc) {
    strcpy (modelfile, argv[i+1]);
  }
  if(learn_parm->svm_iter_to_shrink == -9999) {
    if(kernel_parm->kernel_type == LINEAR) 
      learn_parm->svm_iter_to_shrink=2;
    else
      learn_parm->svm_iter_to_shrink=100;
  }
  if(strcmp(type,"c")==0) {
    learn_parm->type=CLASSIFICATION;
  }
  else if(strcmp(type,"r")==0) {
    learn_parm->type=REGRESSION;
  }
  else if(strcmp(type,"p")==0) {
    learn_parm->type=RANKING;
  }
  else {
    printf("\nUnknown type '%s': Valid types are 'c' (classification), 'r' regession, and 'p' preference ranking.\n",type);
    wait_any_key();
    print_help();
    exit(0);
  }    
  if((learn_parm->skip_final_opt_check) 
     && (kernel_parm->kernel_type == LINEAR)) {
    printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
    learn_parm->skip_final_opt_check=0;
  }    
  if((learn_parm->skip_final_opt_check) 
     && (learn_parm->remove_inconsistent)) {
    printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
    wait_any_key();
    print_help();
    exit(0);
  }    
  if((learn_parm->svm_maxqpsize<2)) {
    printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n",learn_parm->svm_maxqpsize); 
    wait_any_key();
    print_help();
    exit(0);
  }
  if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
    printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n",learn_parm->svm_maxqpsize); 
    printf("new variables [%ld] entering the working set in each iteration.\n",learn_parm->svm_newvarsinqp); 
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->svm_iter_to_shrink<1) {
    printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",learn_parm->svm_iter_to_shrink);
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->svm_c<0) {
    printf("\nThe C parameter must be greater than zero!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->transduction_posratio>1) {
    printf("\nThe fraction of unlabeled examples to classify as positives must\n");
    printf("be less than 1.0 !!!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->svm_costratio<=0) {
    printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->epsilon_crit<=0) {
    printf("\nThe epsilon parameter must be greater than zero!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->rho<0) {
    printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
    printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
    printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
    printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
    printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
    printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
    wait_any_key();
    print_help();
    exit(0);
  }
}

void wait_any_key()
{
  printf("\n(more)\n");
  (void)getc(stdin);
}

void print_help()
{
  printf("\nTree Kernels in SVM-light %s : SVM Learning module %s\n",VERSION,VERSION_DATE);
  printf("by Alessandro Moschitti, moschitti@info.uniroma2.it\n");
  printf("University of Rome \"Tor Vergata\"\n\n");

  copyright_notice();
  printf("   usage: svm_learn [options] example_file model_file\n\n");
  printf("Arguments:\n");
  printf("         example_file-> file with training data\n");
  printf("         model_file  -> file to store learned decision rule in\n");

  printf("General options:\n");
  printf("         -?          -> this help\n");
  printf("         -v [0..3]   -> verbosity level (default 1)\n");
  printf("Learning options:\n");
  printf("         -z {c,r,p}  -> select between classification (c), regression (r),\n");
  printf("                        and preference ranking (p) (default classification)\n");
  printf("         -c float    -> C: trade-off between training error\n");
  printf("                        and margin (default [avg. x*x]^-1)\n");
  printf("         -w [0..]    -> epsilon width of tube for regression\n");
  printf("                        (default 0.1)\n");
  printf("         -j float    -> Cost: cost-factor, by which training errors on\n");
  printf("                        positive examples outweight errors on negative\n");
  printf("                        examples (default 1) (see [4])\n");
  printf("         -b [0,1]    -> use biased hyperplane (i.e. x*w+b>0) instead\n");
  printf("                        of unbiased hyperplane (i.e. x*w>0) (default 1)\n");
  printf("         -i [0,1]    -> remove inconsistent training examples\n");
  printf("                        and retrain (default 0)\n");
  printf("Performance estimation options:\n");
  printf("         -x [0,1]    -> compute leave-one-out estimates (default 0)\n");
  printf("                        (see [5])\n");
  printf("         -o ]0..2]   -> value of rho for XiAlpha-estimator and for pruning\n");
  printf("                        leave-one-out computation (default 1.0) (see [2])\n");
  printf("         -k [0..100] -> search depth for extended XiAlpha-estimator \n");
  printf("                        (default 0)\n");
  printf("Transduction options (see [3]):\n");
  printf("         -p [0..1]   -> fraction of unlabeled examples to be classified\n");
  printf("                        into the positive class (default is the ratio of\n");
  printf("                        positive and negative examples in the training data)\n");

  printf("Kernel options:\n");
  printf("         -t int      -> type of kernel function:\n");
  printf("                        0: linear (default)\n");
  printf("                        1: polynomial (s a*b+c)^d\n");
  printf("                        2: radial basis function exp(-gamma ||a-b||^2)\n");
  printf("                        3: sigmoid tanh(s a*b + c)\n");
  printf("                        4: user defined kernel from kernel.h\n");

  printf("                        5: combination of forest and vector sets according to W, V, S, C options\n");
  printf("                        11: re-ranking based on trees (each instance must have two trees),\n");
  printf("                        12: re-ranking based on vectors (each instance must have two vectors)\n");
  printf("                        13: re-ranking based on both tree and vectors (each instance must have\n");
  printf("                            two trees and two vectors)  \n");
  printf("         -W [S,A]    -> with an 'S', a tree kernel is applied to the sequence of trees of two input\n");
  printf("                        forests and the results are summed;  \n");
  printf("                     -> with an 'A', a tree kernel is applied to all tree pairs from the two forests\n");
  printf("                        (default 'S')\n");
  printf("         -V [S,A]    -> same as before but regarding sequences of vectors are used (default 'S' and\n");
  printf("                        the type of vector-based kernel is specified by the option -S)\n");
  printf("         -S [0,4]    -> kernel to be used with vectors (default polynomial of degree 3,\n");
  printf("                        i.e. -S = 1 and -d = 3)\n");
  printf("         -C [*,+,T,V]-> combination operator between forests and vectors (default 'T')\n");
  printf("                     -> 'T' only the contribution from trees is used (specified by option -W)\n");
  printf("                     -> 'V' only the contribution from vectors is used (specified by option -V)\n");
  printf("                     -> '+' or '*' sum or multiplication of the contributions from vectors and \n");
  printf("                            trees (default T) \n");
  printf("         -D [0,1]    -> 0, SubTree kernel or 1, SubSet Tree kernels (default 1)\n");
  printf("         -L float    -> decay factor in tree kernel (default 0.4)\n");
  printf("         -S [0,4]    -> kernel to be used with vectors (default polynomial of degree 3, \n");
  printf("                        i.e. -S = 1 and -d = 3)\n");
  printf("         -T float    -> multiplicative constant for the contribution of tree kernels when -C = '+'\n");
  printf("         -N float    -> 0 = no normalization, 1 = tree normalization, 2 = vector normalization and \n");
  printf("                        3 = tree normalization of both trees and vectors. The normalization is applied \n");
  printf("                        to each individual tree or vector (default 3).\n");

  printf("         -u string   -> parameter of user defined kernel\n");
  printf("         -d int      -> parameter d in polynomial kernel\n");
  printf("         -g float    -> parameter gamma in rbf kernel\n");
  printf("         -s float    -> parameter s in sigmoid/poly kernel\n");
  printf("         -r float    -> parameter c in sigmoid/poly kernel\n");
  printf("         -u string   -> parameter of user defined kernel\n");
 
  printf("Optimization options (see [1]):\n");
  printf("         -q [2..]    -> maximum size of QP-subproblems (default 10)\n");
  printf("         -n [2..q]   -> number of new variables entering the working set\n");
  printf("                        in each iteration (default n = q). Set n<q to prevent\n");
  printf("                        zig-zagging.\n");
  printf("         -m [5..]    -> size of cache for kernel evaluations in MB (default 40)\n");
  printf("                        The larger the faster...\n");
  printf("         -e float    -> eps: Allow that error for termination criterion\n");
  printf("                        [y [w*x+b] - 1] >= eps (default 0.001)\n");
  printf("         -h [5..]    -> number of iterations a variable needs to be\n"); 
  printf("                        optimal before considered for shrinking (default 100)\n");
  printf("         -f [0,1]    -> do final optimality check for variables removed\n");
  printf("                        by shrinking. Although this test is usually \n");
  printf("                        positive, there is no guarantee that the optimum\n");
  printf("                        was found if the test is omitted. (default 1)\n");
  printf("Output options:\n");
  printf("         -l string   -> file to write predicted labels of unlabeled\n");
  printf("                        examples into after transductive learning\n");
  printf("         -a string   -> write all alphas to this file after learning\n");
  printf("                        (in the same order as in the training set)\n");
  wait_any_key();
  printf("\nMore details in:\n");
  printf("[1] T. Joachims, Making Large-Scale SVM Learning Practical. Advances in\n");
  printf("    Kernel Methods - Support Vector Learning, B. Schölkopf and C. Burges and\n");
  printf("    A. Smola (ed.), MIT Press, 1999.\n");
  printf("[2] T. Joachims, Estimating the Generalization performance of an SVM\n");
  printf("    Efficiently. International Conference on Machine Learning (ICML), 2000.\n");
  printf("[3] T. Joachims, Transductive Inference for Text Classification using Support\n");
  printf("    Vector Machines. International Conference on Machine Learning (ICML),\n");
  printf("    1999.\n");
  printf("[4] K. Morik, P. Brockhausen, and T. Joachims, Combining statistical learning\n");
  printf("    with a knowledge-based approach - A case study in intensive care  \n");
  printf("    monitoring. International Conference on Machine Learning (ICML), 1999.\n");
  printf("[5] T. Joachims, Learning to Classify Text Using Support Vector\n");
  printf("    Machines: Methods, Theory, and Algorithms. Dissertation, Kluwer,\n");
  printf("    2002.\n\n");
  printf("\nFor Tree-Kernel details:\n");
  printf("[6] A. Moschitti, A study on Convolution Kernels for Shallow Semantic Parsing.\n");
  printf("    In proceedings of the 42-th Conference on Association for Computational\n");
  printf("    Linguistic, (ACL-2004), Barcelona, Spain, 2004.\n\n");
  printf("[7] A. Moschitti, Making tree kernels practical for natural language learning.\n");
  printf("    In Proceedings of the Eleventh International Conference for Computational\n");
  printf("    Linguistics, (EACL-2006), Trento, Italy, 2006.\n\n");
  
}
