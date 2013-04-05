/***********************************************************************/
/*                                                                     */
/*   svm_learn.c                                                       */
/*                                                                     */
/*   Learning module of Support Vector Machine.                        */
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


# include "svm_common.h"
# include "svm_learn.h"


/* interface to QP-solver */
double *optimize_qp(QP *, double *, long, double *, LEARN_PARM *);

/*---------------------------------------------------------------------------*/

/* Learns an SVM classification model based on the training data in
   docs/label. The resulting model is returned in the structure
   model. */

void svm_learn_classification(DOC **docs, double *class, long int
			      totdoc, long int totwords, 
			      LEARN_PARM *learn_parm, 
			      KERNEL_PARM *kernel_parm, 
			      KERNEL_CACHE *kernel_cache, 
			      MODEL *model,
			      double *alpha)
     /* docs:        Training vectors (x-part) */
     /* class:       Training labels (y-part, zero if test example for
                     transduction) */
     /* totdoc:      Number of examples in docs/label */
     /* totwords:    Number of features (i.e. highest feature index) */
     /* learn_parm:  Learning paramenters */
     /* kernel_parm: Kernel paramenters */
     /* kernel_cache:Initialized Cache of size totdoc, if using a kernel. 
                     NULL if linear.*/
     /* model:       Returns learning result (assumed empty before called) */
     /* alpha:       Start values for the alpha variables or NULL
	             pointer. The new alpha values are returned after 
		     optimization if not NULL. Array must be of size totdoc. */
{
  long *inconsistent,i,*label;
  long inconsistentnum;
  long misclassified,upsupvecnum;
  double loss,model_length,example_length;
  double maxdiff,*lin,*a,*c;
  long runtime_start,runtime_end;
  long iterations;
  long *unlabeled,transduction;
  long heldout;
  long loo_count=0,loo_count_pos=0,loo_count_neg=0,trainpos=0,trainneg=0;
  long loocomputed=0,runtime_start_loo=0,runtime_start_xa=0;
  double heldout_c=0,r_delta_sq=0,r_delta,r_delta_avg;
  long *index,*index2dnum;
  double *weights;
  CFLOAT *aicache;  /* buffer to keep one row of hessian */

  double *xi_fullset; /* buffer for storing xi on full sample in loo */
  double *a_fullset;  /* buffer for storing alpha on full sample in loo */
  TIMING timing_profile;
  SHRINK_STATE shrink_state;

  runtime_start=get_runtime();
  timing_profile.time_kernel=0;
  timing_profile.time_opti=0;
  timing_profile.time_shrink=0;
  timing_profile.time_update=0;
  timing_profile.time_model=0;
  timing_profile.time_check=0;
  timing_profile.time_select=0;
  kernel_cache_statistic=0;

  learn_parm->totwords=totwords;

  /* make sure -n value is reasonable */
  if((learn_parm->svm_newvarsinqp < 2) 
     || (learn_parm->svm_newvarsinqp > learn_parm->svm_maxqpsize)) {
    learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize;
  }

  init_shrink_state(&shrink_state,totdoc,(long)MAXSHRINK);

  label = (long *)my_malloc(sizeof(long)*totdoc);
  inconsistent = (long *)my_malloc(sizeof(long)*totdoc);
  unlabeled = (long *)my_malloc(sizeof(long)*totdoc);
  c = (double *)my_malloc(sizeof(double)*totdoc);
  a = (double *)my_malloc(sizeof(double)*totdoc);
  a_fullset = (double *)my_malloc(sizeof(double)*totdoc);
  xi_fullset = (double *)my_malloc(sizeof(double)*totdoc);
  lin = (double *)my_malloc(sizeof(double)*totdoc);
  learn_parm->svm_cost = (double *)my_malloc(sizeof(double)*totdoc);
  model->supvec = (DOC **)my_malloc(sizeof(DOC *)*(totdoc+2));
  model->alpha = (double *)my_malloc(sizeof(double)*(totdoc+2));
  model->index = (long *)my_malloc(sizeof(long)*(totdoc+2));

  model->at_upper_bound=0;
  model->b=0;	       
  model->supvec[0]=0;  /* element 0 reserved and empty for now */
  model->alpha[0]=0;
  model->lin_weights=NULL;
  model->totwords=totwords;
  model->totdoc=totdoc;
  model->kernel_parm=(*kernel_parm);
  model->sv_num=1;
  model->loo_error=-1;
  model->loo_recall=-1;
  model->loo_precision=-1;
  model->xa_error=-1;
  model->xa_recall=-1;
  model->xa_precision=-1;
  inconsistentnum=0;
  transduction=0;

  r_delta=estimate_r_delta(docs,totdoc,kernel_parm);
  r_delta_sq=r_delta*r_delta;

  r_delta_avg=estimate_r_delta_average(docs,totdoc,kernel_parm);
  if(learn_parm->svm_c == 0.0) {  /* default value for C */
    learn_parm->svm_c=1.0/(r_delta_avg*r_delta_avg);
    if(verbosity>=1) 
      printf("Setting default regularization parameter C=%.4f\n",
	     learn_parm->svm_c);
  }

  learn_parm->eps=-1.0;      /* equivalent regression epsilon for
				classification */

  for(i=0;i<totdoc;i++) {    /* various inits */
    docs[i]->docnum=i;
    inconsistent[i]=0;
    a[i]=0;
    lin[i]=0;
    c[i]=0.0;
    unlabeled[i]=0;
    if(class[i] == 0) {
      unlabeled[i]=1;
      label[i]=0;
      transduction=1;
    }
    if(class[i] > 0) {
      learn_parm->svm_cost[i]=learn_parm->svm_c*learn_parm->svm_costratio*
	docs[i]->costfactor;
      label[i]=1;
      trainpos++;
    }
    else if(class[i] < 0) {
      learn_parm->svm_cost[i]=learn_parm->svm_c*docs[i]->costfactor;
      label[i]=-1;
      trainneg++;
    }
    else {
      learn_parm->svm_cost[i]=0;
    }
  }
  if(verbosity>=2) {
    printf("%ld positive, %ld negative, and %ld unlabeled examples.\n",trainpos,trainneg,totdoc-trainpos-trainneg); fflush(stdout);
  }

  /* caching makes no sense for linear kernel */
  if(kernel_parm->kernel_type == LINEAR) {
    kernel_cache = NULL;   
  } 

  /* compute starting state for initial alpha values */
  if(alpha) {
    if(verbosity>=1) {
      printf("Computing starting state..."); fflush(stdout);
    }
    index = (long *)my_malloc(sizeof(long)*totdoc);
    index2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
    weights=(double *)my_malloc(sizeof(double)*(totwords+1));
    aicache = (CFLOAT *)my_malloc(sizeof(CFLOAT)*totdoc);
    for(i=0;i<totdoc;i++) {    /* create full index and clip alphas */
      index[i]=1;
      alpha[i]=fabs(alpha[i]);
      if(alpha[i]<0) alpha[i]=0;
      if(alpha[i]>learn_parm->svm_cost[i]) alpha[i]=learn_parm->svm_cost[i];
    }
    if(kernel_parm->kernel_type != LINEAR) {
      for(i=0;i<totdoc;i++)     /* fill kernel cache with unbounded SV */
	if((alpha[i]>0) && (alpha[i]<learn_parm->svm_cost[i]) 
	   && (kernel_cache_space_available(kernel_cache))) 
	  cache_kernel_row(kernel_cache,docs,i,kernel_parm);
      for(i=0;i<totdoc;i++)     /* fill rest of kernel cache with bounded SV */
	if((alpha[i]==learn_parm->svm_cost[i]) 
	   && (kernel_cache_space_available(kernel_cache))) 
	  cache_kernel_row(kernel_cache,docs,i,kernel_parm);
    }
    (void)compute_index(index,totdoc,index2dnum);
    update_linear_component(docs,label,index2dnum,alpha,a,index2dnum,totdoc,
			    totwords,kernel_parm,kernel_cache,lin,aicache,
			    weights);
    (void)calculate_svm_model(docs,label,unlabeled,lin,alpha,a,c,
			      learn_parm,index2dnum,index2dnum,model);
    for(i=0;i<totdoc;i++) {    /* copy initial alphas */
      a[i]=alpha[i];
    }
    free(index);
    free(index2dnum);
    free(weights);
    free(aicache);
    if(verbosity>=1) {
      printf("done.\n");  fflush(stdout);
    }   
  } 

  if(transduction) {
    learn_parm->svm_iter_to_shrink=99999999;
    if(verbosity >= 1)
      printf("\nDeactivating Shrinking due to an incompatibility with the transductive \nlearner in the current version.\n\n");
  }

  if(transduction && learn_parm->compute_loo) {
    learn_parm->compute_loo=0;
    if(verbosity >= 1)
      printf("\nCannot compute leave-one-out estimates for transductive learner.\n\n");
  }    

  if(learn_parm->remove_inconsistent && learn_parm->compute_loo) {
    learn_parm->compute_loo=0;
    printf("\nCannot compute leave-one-out estimates when removing inconsistent examples.\n\n");
  }    

  if(learn_parm->compute_loo && ((trainpos == 1) || (trainneg == 1))) {
    learn_parm->compute_loo=0;
    printf("\nCannot compute leave-one-out with only one example in one class.\n\n");
  }    


  if(verbosity==1) {
    printf("Optimizing"); fflush(stdout);
  }

  /* train the svm */
  iterations=optimize_to_convergence(docs,label,totdoc,totwords,learn_parm,
				     kernel_parm,kernel_cache,&shrink_state,model,
				     inconsistent,unlabeled,a,lin,
				     c,&timing_profile,
				     &maxdiff,(long)-1,
				     (long)1);
  
  if(verbosity>=1) {
    if(verbosity==1) printf("done. (%ld iterations)\n",iterations);

    misclassified=0;
    for(i=0;(i<totdoc);i++) { /* get final statistic */
      if((lin[i]-model->b)*(double)label[i] <= 0.0) 
	misclassified++;
    }

    printf("Optimization finished (%ld misclassified, maxdiff=%.5f).\n",
	   misclassified,maxdiff); 

    runtime_end=get_runtime();
    if(verbosity>=2) {
      printf("Runtime in cpu-seconds: %.2f (%.2f%% for kernel/%.2f%% for optimizer/%.2f%% for final/%.2f%% for update/%.2f%% for model/%.2f%% for check/%.2f%% for select)\n",
        ((float)runtime_end-(float)runtime_start)/100.0,
        (100.0*timing_profile.time_kernel)/(float)(runtime_end-runtime_start),
	(100.0*timing_profile.time_opti)/(float)(runtime_end-runtime_start),
	(100.0*timing_profile.time_shrink)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_update)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_model)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_check)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_select)/(float)(runtime_end-runtime_start));
    }
    else {
      printf("Runtime in cpu-seconds: %.2f\n",
	     (runtime_end-runtime_start)/100.0);
    }

    if(learn_parm->remove_inconsistent) {	  
      inconsistentnum=0;
      for(i=0;i<totdoc;i++) 
	if(inconsistent[i]) 
	  inconsistentnum++;
      printf("Number of SV: %ld (plus %ld inconsistent examples)\n",
	     model->sv_num-1,inconsistentnum);
    }
    else {
      upsupvecnum=0;
      for(i=1;i<model->sv_num;i++) {
	if(fabs(model->alpha[i]) >= 
	   (learn_parm->svm_cost[(model->supvec[i])->docnum]-
	    learn_parm->epsilon_a)) 
	  upsupvecnum++;
      }
      printf("Number of SV: %ld (including %ld at upper bound)\n",
	     model->sv_num-1,upsupvecnum);
    }
    
    if((verbosity>=1) && (!learn_parm->skip_final_opt_check)) {
      loss=0;
      model_length=0; 
      for(i=0;i<totdoc;i++) {
	if((lin[i]-model->b)*(double)label[i] < 1.0-learn_parm->epsilon_crit)
	  loss+=1.0-(lin[i]-model->b)*(double)label[i];
	model_length+=a[i]*label[i]*lin[i];
      }
      model_length=sqrt(model_length);
      fprintf(stdout,"L1 loss: loss=%.5f\n",loss);
      fprintf(stdout,"Norm of weight vector: |w|=%.5f\n",model_length);
      example_length=estimate_sphere(model,kernel_parm); 
      fprintf(stdout,"Norm of longest example vector: |x|=%.5f\n",
	      length_of_longest_document_vector(docs,totdoc,kernel_parm));
      fprintf(stdout,"Estimated VCdim of classifier: VCdim<=%.5f\n",
	      estimate_margin_vcdim(model,model_length,example_length,
				    kernel_parm));
      if((!learn_parm->remove_inconsistent) && (!transduction)) {
	runtime_start_xa=get_runtime();
	if(verbosity>=1) {
	  printf("Computing XiAlpha-estimates..."); fflush(stdout);
	}
	compute_xa_estimates(model,label,unlabeled,totdoc,docs,lin,a,
			     kernel_parm,learn_parm,&(model->xa_error),
			     &(model->xa_recall),&(model->xa_precision));
	if(verbosity>=1) {
	  printf("done\n");
	}
	printf("Runtime for XiAlpha-estimates in cpu-seconds: %.2f\n",
	       (get_runtime()-runtime_start_xa)/100.0);
	
	fprintf(stdout,"XiAlpha-estimate of the error: error<=%.2f%% (rho=%.2f,depth=%ld)\n",
		model->xa_error,learn_parm->rho,learn_parm->xa_depth);
	fprintf(stdout,"XiAlpha-estimate of the recall: recall=>%.2f%% (rho=%.2f,depth=%ld)\n",
		model->xa_recall,learn_parm->rho,learn_parm->xa_depth);
	fprintf(stdout,"XiAlpha-estimate of the precision: precision=>%.2f%% (rho=%.2f,depth=%ld)\n",
		model->xa_precision,learn_parm->rho,learn_parm->xa_depth);
      }
      else if(!learn_parm->remove_inconsistent) {
	estimate_transduction_quality(model,label,unlabeled,totdoc,docs,lin);
      }
    }
    if(verbosity>=1) {
      printf("Number of kernel evaluations: %ld\n",kernel_cache_statistic);
    }
  }


  /* leave-one-out testing starts now */
  if(learn_parm->compute_loo) {
    /* save results of training on full dataset for leave-one-out */
    runtime_start_loo=get_runtime();
    for(i=0;i<totdoc;i++) {
      xi_fullset[i]=1.0-((lin[i]-model->b)*(double)label[i]);
      if(xi_fullset[i]<0) xi_fullset[i]=0;
      a_fullset[i]=a[i];
    }
    if(verbosity>=1) {
      printf("Computing leave-one-out");
    }
    
    /* repeat this loop for every held-out example */
    for(heldout=0;(heldout<totdoc);heldout++) {
      if(learn_parm->rho*a_fullset[heldout]*r_delta_sq+xi_fullset[heldout]
	 < 1.0) { 
	/* guaranteed to not produce a leave-one-out error */
	if(verbosity==1) {
	  printf("+"); fflush(stdout); 
	}
      }
      else if(xi_fullset[heldout] > 1.0) {
	/* guaranteed to produce a leave-one-out error */
	loo_count++;
	if(label[heldout] > 0)  loo_count_pos++; else loo_count_neg++;
	if(verbosity==1) {
	  printf("-"); fflush(stdout); 
	}
      }
      else {
	loocomputed++;
	heldout_c=learn_parm->svm_cost[heldout]; /* set upper bound to zero */
	learn_parm->svm_cost[heldout]=0;
	/* make sure heldout example is not currently  */
	/* shrunk away. Assumes that lin is up to date! */
	shrink_state.active[heldout]=1;  
	if(verbosity>=2) 
	  printf("\nLeave-One-Out test on example %ld\n",heldout);
	if(verbosity>=1) {
	  printf("(?[%ld]",heldout); fflush(stdout); 
	}
	
	optimize_to_convergence(docs,label,totdoc,totwords,learn_parm,
				kernel_parm,
				kernel_cache,&shrink_state,model,inconsistent,unlabeled,
				a,lin,c,&timing_profile,
				&maxdiff,heldout,(long)2);

	/* printf("%.20f\n",(lin[heldout]-model->b)*(double)label[heldout]); */

	if(((lin[heldout]-model->b)*(double)label[heldout]) <= 0.0) { 
	  loo_count++;                            /* there was a loo-error */
	  if(label[heldout] > 0)  loo_count_pos++; else loo_count_neg++;
	  if(verbosity>=1) {
	    printf("-)"); fflush(stdout); 
	  }
	}
	else {
	  if(verbosity>=1) {
	    printf("+)"); fflush(stdout); 
	  }
	}
	/* now we need to restore the original data set*/
	learn_parm->svm_cost[heldout]=heldout_c; /* restore upper bound */
      }
    } /* end of leave-one-out loop */


    if(verbosity>=1) {
      printf("\nRetrain on full problem"); fflush(stdout); 
    }
    optimize_to_convergence(docs,label,totdoc,totwords,learn_parm,
			    kernel_parm,
			    kernel_cache,&shrink_state,model,inconsistent,unlabeled,
			    a,lin,c,&timing_profile,
			    &maxdiff,(long)-1,(long)1);
    if(verbosity >= 1) 
      printf("done.\n");
    
    
    /* after all leave-one-out computed */
    model->loo_error=100.0*loo_count/(double)totdoc;
    model->loo_recall=(1.0-(double)loo_count_pos/(double)trainpos)*100.0;
    model->loo_precision=(trainpos-loo_count_pos)/
      (double)(trainpos-loo_count_pos+loo_count_neg)*100.0;
    if(verbosity >= 1) {
      fprintf(stdout,"Leave-one-out estimate of the error: error=%.2f%%\n",
	      model->loo_error);
      fprintf(stdout,"Leave-one-out estimate of the recall: recall=%.2f%%\n",
	      model->loo_recall);
      fprintf(stdout,"Leave-one-out estimate of the precision: precision=%.2f%%\n",
	      model->loo_precision);
      fprintf(stdout,"Actual leave-one-outs computed:  %ld (rho=%.2f)\n",
	      loocomputed,learn_parm->rho);
      printf("Runtime for leave-one-out in cpu-seconds: %.2f\n",
	     (double)(get_runtime()-runtime_start_loo)/100.0);
    }
  }
    
  if(learn_parm->alphafile[0])
    write_alphas(learn_parm->alphafile,a,label,totdoc);
  
  shrink_state_cleanup(&shrink_state);
  free(label);
  free(inconsistent);
  free(unlabeled);
  free(c);
  free(a);
  free(a_fullset);
  free(xi_fullset);
  free(lin);
  free(learn_parm->svm_cost);
}


/* Learns an SVM regression model based on the training data in
   docs/label. The resulting model is returned in the structure
   model. */

void svm_learn_regression(DOC **docs, double *value, long int totdoc, 
			  long int totwords, LEARN_PARM *learn_parm, 
			  KERNEL_PARM *kernel_parm, 
			  KERNEL_CACHE **kernel_cache, MODEL *model)
     /* docs:        Training vectors (x-part) */
     /* class:       Training value (y-part) */
     /* totdoc:      Number of examples in docs/label */
     /* totwords:    Number of features (i.e. highest feature index) */
     /* learn_parm:  Learning paramenters */
     /* kernel_parm: Kernel paramenters */
     /* kernel_cache:Initialized Cache, if using a kernel. NULL if
                     linear. Note that it will be free'd and reassigned */
     /* model:       Returns learning result (assumed empty before called) */
{
  long *inconsistent,i,j;
  long inconsistentnum;
  long upsupvecnum;
  double loss,model_length,example_length;
  double maxdiff,*lin,*a,*c;
  long runtime_start,runtime_end;
  long iterations,kernel_cache_size;
  long *unlabeled;
  double r_delta_sq=0,r_delta,r_delta_avg;
  double *xi_fullset; /* buffer for storing xi on full sample in loo */
  double *a_fullset;  /* buffer for storing alpha on full sample in loo */
  TIMING timing_profile;
  SHRINK_STATE shrink_state;
  DOC **docs_org;
  long *label;

  /* set up regression problem in standard form */
  docs_org=docs;
  docs = (DOC **)my_malloc(sizeof(DOC)*2*totdoc);
  label = (long *)my_malloc(sizeof(long)*2*totdoc);
  c = (double *)my_malloc(sizeof(double)*2*totdoc);
  for(i=0;i<totdoc;i++) {   
    j=2*totdoc-1-i;
    docs[i]=create_example(i,0,0,docs_org[i]->costfactor,docs_org[i]->fvec);
    label[i]=+1;
    c[i]=value[i];
    docs[j]=create_example(j,0,0,docs_org[i]->costfactor,docs_org[i]->fvec);
    label[j]=-1;
    c[j]=value[i];
  }
  totdoc*=2;

  /* need to get a bigger kernel cache */
  if(*kernel_cache) {
    kernel_cache_size=(*kernel_cache)->buffsize*sizeof(CFLOAT)/(1024*1024);
    kernel_cache_cleanup(*kernel_cache);
    (*kernel_cache)=kernel_cache_init(totdoc,kernel_cache_size);
  }

  runtime_start=get_runtime();
  timing_profile.time_kernel=0;
  timing_profile.time_opti=0;
  timing_profile.time_shrink=0;
  timing_profile.time_update=0;
  timing_profile.time_model=0;
  timing_profile.time_check=0;
  timing_profile.time_select=0;
  kernel_cache_statistic=0;

  learn_parm->totwords=totwords;

  /* make sure -n value is reasonable */
  if((learn_parm->svm_newvarsinqp < 2) 
     || (learn_parm->svm_newvarsinqp > learn_parm->svm_maxqpsize)) {
    learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize;
  }

  init_shrink_state(&shrink_state,totdoc,(long)MAXSHRINK);

  inconsistent = (long *)my_malloc(sizeof(long)*totdoc);
  unlabeled = (long *)my_malloc(sizeof(long)*totdoc);
  a = (double *)my_malloc(sizeof(double)*totdoc);
  a_fullset = (double *)my_malloc(sizeof(double)*totdoc);
  xi_fullset = (double *)my_malloc(sizeof(double)*totdoc);
  lin = (double *)my_malloc(sizeof(double)*totdoc);
  learn_parm->svm_cost = (double *)my_malloc(sizeof(double)*totdoc);
  model->supvec = (DOC **)my_malloc(sizeof(DOC *)*(totdoc+2));
  model->alpha = (double *)my_malloc(sizeof(double)*(totdoc+2));
  model->index = (long *)my_malloc(sizeof(long)*(totdoc+2));

  model->at_upper_bound=0;
  model->b=0;	       
  model->supvec[0]=0;  /* element 0 reserved and empty for now */
  model->alpha[0]=0;
  model->lin_weights=NULL;
  model->totwords=totwords;
  model->totdoc=totdoc;
  model->kernel_parm=(*kernel_parm);
  model->sv_num=1;
  model->loo_error=-1;
  model->loo_recall=-1;
  model->loo_precision=-1;
  model->xa_error=-1;
  model->xa_recall=-1;
  model->xa_precision=-1;
  inconsistentnum=0;

  r_delta=estimate_r_delta(docs,totdoc,kernel_parm);
  r_delta_sq=r_delta*r_delta;

  r_delta_avg=estimate_r_delta_average(docs,totdoc,kernel_parm);
  if(learn_parm->svm_c == 0.0) {  /* default value for C */
    learn_parm->svm_c=1.0/(r_delta_avg*r_delta_avg);
    if(verbosity>=1) 
      printf("Setting default regularization parameter C=%.4f\n",
	     learn_parm->svm_c);
  }

  for(i=0;i<totdoc;i++) {    /* various inits */
    inconsistent[i]=0;
    a[i]=0;
    lin[i]=0;
    unlabeled[i]=0;
    if(label[i] > 0) {
      learn_parm->svm_cost[i]=learn_parm->svm_c*learn_parm->svm_costratio*
	docs[i]->costfactor;
    }
    else if(label[i] < 0) {
      learn_parm->svm_cost[i]=learn_parm->svm_c*docs[i]->costfactor;
    }
  }

  /* caching makes no sense for linear kernel */
  if((kernel_parm->kernel_type == LINEAR) && (*kernel_cache)) {
    printf("WARNING: Using a kernel cache for linear case will slow optimization down!\n");
  } 

  if(verbosity==1) {
    printf("Optimizing"); fflush(stdout);
  }

  /* train the svm */
  iterations=optimize_to_convergence(docs,label,totdoc,totwords,learn_parm,
				     kernel_parm,*kernel_cache,&shrink_state,
				     model,inconsistent,unlabeled,a,lin,c,
				     &timing_profile,&maxdiff,(long)-1,
				     (long)1);
  
  if(verbosity>=1) {
    if(verbosity==1) printf("done. (%ld iterations)\n",iterations);

    printf("Optimization finished (maxdiff=%.5f).\n",maxdiff); 

    runtime_end=get_runtime();
    if(verbosity>=2) {
      printf("Runtime in cpu-seconds: %.2f (%.2f%% for kernel/%.2f%% for optimizer/%.2f%% for final/%.2f%% for update/%.2f%% for model/%.2f%% for check/%.2f%% for select)\n",
        ((float)runtime_end-(float)runtime_start)/100.0,
        (100.0*timing_profile.time_kernel)/(float)(runtime_end-runtime_start),
	(100.0*timing_profile.time_opti)/(float)(runtime_end-runtime_start),
	(100.0*timing_profile.time_shrink)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_update)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_model)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_check)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_select)/(float)(runtime_end-runtime_start));
    }
    else {
      printf("Runtime in cpu-seconds: %.2f\n",
	     (runtime_end-runtime_start)/100.0);
    }

    if(learn_parm->remove_inconsistent) {	  
      inconsistentnum=0;
      for(i=0;i<totdoc;i++) 
	if(inconsistent[i]) 
	  inconsistentnum++;
      printf("Number of SV: %ld (plus %ld inconsistent examples)\n",
	     model->sv_num-1,inconsistentnum);
    }
    else {
      upsupvecnum=0;
      for(i=1;i<model->sv_num;i++) {
	if(fabs(model->alpha[i]) >= 
	   (learn_parm->svm_cost[(model->supvec[i])->docnum]-
	    learn_parm->epsilon_a)) 
	  upsupvecnum++;
      }
      printf("Number of SV: %ld (including %ld at upper bound)\n",
	     model->sv_num-1,upsupvecnum);
    }
    
    if((verbosity>=1) && (!learn_parm->skip_final_opt_check)) {
      loss=0;
      model_length=0; 
      for(i=0;i<totdoc;i++) {
	if((lin[i]-model->b)*(double)label[i] < (-learn_parm->eps+(double)label[i]*c[i])-learn_parm->epsilon_crit)
	  loss+=-learn_parm->eps+(double)label[i]*c[i]-(lin[i]-model->b)*(double)label[i];
	model_length+=a[i]*label[i]*lin[i];
      }
      model_length=sqrt(model_length);
      fprintf(stdout,"L1 loss: loss=%.5f\n",loss);
      fprintf(stdout,"Norm of weight vector: |w|=%.5f\n",model_length);
      example_length=estimate_sphere(model,kernel_parm); 
      fprintf(stdout,"Norm of longest example vector: |x|=%.5f\n",
	      length_of_longest_document_vector(docs,totdoc,kernel_parm));
    }
    if(verbosity>=1) {
      printf("Number of kernel evaluations: %ld\n",kernel_cache_statistic);
    }
  }
    
  if(learn_parm->alphafile[0])
    write_alphas(learn_parm->alphafile,a,label,totdoc);

  /* this makes sure the model we return does not contain pointers to the 
     temporary documents */
  for(i=1;i<model->sv_num;i++) { 
    j=model->supvec[i]->docnum;
    if(j >= (totdoc/2)) {
      j=totdoc-j-1;
    }
    model->supvec[i]=docs_org[j];
  }
  
  shrink_state_cleanup(&shrink_state);
  for(i=0;i<totdoc;i++)
    free_example(docs[i],0);
  free(docs);
  free(label);
  free(inconsistent);
  free(unlabeled);
  free(c);
  free(a);
  free(a_fullset);
  free(xi_fullset);
  free(lin);
  free(learn_parm->svm_cost);
}

void svm_learn_ranking(DOC **docs, double *rankvalue, long int totdoc, 
		       long int totwords, LEARN_PARM *learn_parm, 
		       KERNEL_PARM *kernel_parm, KERNEL_CACHE **kernel_cache, 
		       MODEL *model)
     /* docs:        Training vectors (x-part) */
     /* rankvalue:   Training target values that determine the ranking */
     /* totdoc:      Number of examples in docs/label */
     /* totwords:    Number of features (i.e. highest feature index) */
     /* learn_parm:  Learning paramenters */
     /* kernel_parm: Kernel paramenters */
     /* kernel_cache:Initialized pointer to Cache of size 1*totdoc, if 
	             using a kernel. NULL if linear. NOTE: Cache is 
                     getting reinitialized in this function */
     /* model:       Returns learning result (assumed empty before called) */
{
  DOC **docdiff;
  long i,j,k,totpair,kernel_cache_size;
  double *target,*alpha,cost;
  long *greater,*lesser;
  MODEL *pairmodel;
  SVECTOR *flow,*fhigh;

  totpair=0;
  for(i=0;i<totdoc;i++) {
    for(j=i+1;j<totdoc;j++) {
      if((docs[i]->queryid==docs[j]->queryid) && (rankvalue[i] != rankvalue[j])) {
	totpair++;
      }
    }
  }

  printf("Constructing %ld rank constraints...",totpair); fflush(stdout);
  docdiff=(DOC **)my_malloc(sizeof(DOC)*totpair);
  target=(double *)my_malloc(sizeof(double)*totpair); 
  greater=(long *)my_malloc(sizeof(long)*totpair); 
  lesser=(long *)my_malloc(sizeof(long)*totpair); 

  k=0;
  for(i=0;i<totdoc;i++) {
    for(j=i+1;j<totdoc;j++) {
      if(docs[i]->queryid == docs[j]->queryid) {
	cost=(docs[i]->costfactor+docs[j]->costfactor)/2.0;
	if(rankvalue[i] > rankvalue[j]) {
	  if(kernel_parm->kernel_type == LINEAR)
	    docdiff[k]=create_example(k,0,0,cost,
				      sub_ss(docs[i]->fvec,docs[j]->fvec));
	  else {
	    flow=copy_svector(docs[j]->fvec);
	    flow->factor=-1.0;
	    flow->next=NULL;
	    fhigh=copy_svector(docs[i]->fvec);
	    fhigh->factor=1.0;
	    fhigh->next=flow;
	    docdiff[k]=create_example(k,0,0,cost,fhigh);
	  }
	  target[k]=1;
	  greater[k]=i;
	  lesser[k]=j;
	  k++;
	}
	else if(rankvalue[i] < rankvalue[j]) {
	  if(kernel_parm->kernel_type == LINEAR)
	    docdiff[k]=create_example(k,0,0,cost,
				      sub_ss(docs[i]->fvec,docs[j]->fvec));
	  else {
	    flow=copy_svector(docs[j]->fvec);
	    flow->factor=-1.0;
	    flow->next=NULL;
	    fhigh=copy_svector(docs[i]->fvec);
	    fhigh->factor=1.0;
	    fhigh->next=flow;
	    docdiff[k]=create_example(k,0,0,cost,fhigh);
	  }
	  target[k]=-1;
	  greater[k]=i;
	  lesser[k]=j;
	  k++;
	}
      }
    }
  }
  printf("done.\n"); fflush(stdout);

  /* need to get a bigger kernel cache */
  if(*kernel_cache) {
    kernel_cache_size=(*kernel_cache)->buffsize*sizeof(CFLOAT)/(1024*1024);
    kernel_cache_cleanup(*kernel_cache);
    (*kernel_cache)=kernel_cache_init(totpair,kernel_cache_size);
  }

  /* must use unbiased hyperplane on difference vectors */
  learn_parm->biased_hyperplane=0;
  pairmodel=(MODEL *)my_malloc(sizeof(MODEL));
  svm_learn_classification(docdiff,target,totpair,totwords,learn_parm,
			   kernel_parm,(*kernel_cache),pairmodel,NULL);

  /* Transfer the result into a more compact model. If you would like
     to output the original model on pairs of documents, see below. */
  alpha=(double *)my_malloc(sizeof(double)*totdoc); 
  for(i=0;i<totdoc;i++) {
    alpha[i]=0;
  }
  for(i=1;i<pairmodel->sv_num;i++) {
    alpha[lesser[(pairmodel->supvec[i])->docnum]]-=pairmodel->alpha[i];
    alpha[greater[(pairmodel->supvec[i])->docnum]]+=pairmodel->alpha[i];
  }
  model->supvec = (DOC **)my_malloc(sizeof(DOC *)*(totdoc+2));
  model->alpha = (double *)my_malloc(sizeof(double)*(totdoc+2));
  model->index = (long *)my_malloc(sizeof(long)*(totdoc+2));
  model->supvec[0]=0;  /* element 0 reserved and empty for now */
  model->alpha[0]=0;
  model->sv_num=1;
  for(i=0;i<totdoc;i++) {
    if(alpha[i]) {
      model->supvec[model->sv_num]=docs[i];
      model->alpha[model->sv_num]=alpha[i];
      model->index[i]=model->sv_num;
      model->sv_num++;
    }
    else {
      model->index[i]=-1;
    }
  }
  model->at_upper_bound=0;
  model->b=0;	       
  model->lin_weights=NULL;
  model->totwords=totwords;
  model->totdoc=totdoc;
  model->kernel_parm=(*kernel_parm);
  model->loo_error=-1;
  model->loo_recall=-1;
  model->loo_precision=-1;
  model->xa_error=-1;
  model->xa_recall=-1;
  model->xa_precision=-1;

  free(alpha);
  free(greater);
  free(lesser);
  free(target);

  /* If you would like to output the original model on pairs of
     document, replace the following lines with '(*model)=(*pairmodel);' */
  for(i=0;i<totpair;i++)
    free_example(docdiff[i],1);
  free(docdiff);
  free_model(pairmodel,0);
}


/* The following solves a freely defined and given set of
   inequalities. The optimization problem is of the following form:

   min 0.5 w*w + C sum_i C_i \xi_i
   s.t. x_i * w > rhs_i - \xi_i

   This corresponds to the -z o option. */

void svm_learn_optimization(DOC **docs, double *rhs, long int
			    totdoc, long int totwords, 
			    LEARN_PARM *learn_parm, 
			    KERNEL_PARM *kernel_parm, 
			    KERNEL_CACHE *kernel_cache, MODEL *model,
			    double *alpha)
     /* docs:        Left-hand side of inequalities (x-part) */
     /* rhs:         Right-hand side of inequalities */
     /* totdoc:      Number of examples in docs/label */
     /* totwords:    Number of features (i.e. highest feature index) */
     /* learn_parm:  Learning paramenters */
     /* kernel_parm: Kernel paramenters */
     /* kernel_cache:Initialized Cache of size 1*totdoc, if using a kernel. 
                     NULL if linear.*/
     /* model:       Returns solution as SV expansion (assumed empty before called) */
     /* alpha:       Start values for the alpha variables or NULL
	             pointer. The new alpha values are returned after 
		     optimization if not NULL. Array must be of size totdoc. */
{
  long i,*label;
  long misclassified,upsupvecnum;
  double loss,model_length,example_length;
  double maxdiff,*lin,*a,*c;
  long runtime_start,runtime_end;
  long iterations,maxslackid,svsetnum;
  long *unlabeled,*inconsistent;
  double r_delta_sq=0,r_delta,r_delta_avg;
  long *index,*index2dnum;
  double *weights,*slack,*alphaslack;
  CFLOAT *aicache;  /* buffer to keep one row of hessian */

  TIMING timing_profile;
  SHRINK_STATE shrink_state;

  runtime_start=get_runtime();
  timing_profile.time_kernel=0;
  timing_profile.time_opti=0;
  timing_profile.time_shrink=0;
  timing_profile.time_update=0;
  timing_profile.time_model=0;
  timing_profile.time_check=0;
  timing_profile.time_select=0;
  kernel_cache_statistic=0;

  learn_parm->totwords=totwords;

  /* make sure -n value is reasonable */
  if((learn_parm->svm_newvarsinqp < 2) 
     || (learn_parm->svm_newvarsinqp > learn_parm->svm_maxqpsize)) {
    learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize;
  }

  init_shrink_state(&shrink_state,totdoc,(long)MAXSHRINK);

  label = (long *)my_malloc(sizeof(long)*totdoc);
  unlabeled = (long *)my_malloc(sizeof(long)*totdoc);
  inconsistent = (long *)my_malloc(sizeof(long)*totdoc);
  c = (double *)my_malloc(sizeof(double)*totdoc);
  a = (double *)my_malloc(sizeof(double)*totdoc);
  lin = (double *)my_malloc(sizeof(double)*totdoc);
  learn_parm->svm_cost = (double *)my_malloc(sizeof(double)*totdoc);
  model->supvec = (DOC **)my_malloc(sizeof(DOC *)*(totdoc+2));
  model->alpha = (double *)my_malloc(sizeof(double)*(totdoc+2));
  model->index = (long *)my_malloc(sizeof(long)*(totdoc+2));

  model->at_upper_bound=0;
  model->b=0;	       
  model->supvec[0]=0;  /* element 0 reserved and empty for now */
  model->alpha[0]=0;
  model->lin_weights=NULL;
  model->totwords=totwords;
  model->totdoc=totdoc;
  model->kernel_parm=(*kernel_parm);
  model->sv_num=1;
  model->loo_error=-1;
  model->loo_recall=-1;
  model->loo_precision=-1;
  model->xa_error=-1;
  model->xa_recall=-1;
  model->xa_precision=-1;

  r_delta=estimate_r_delta(docs,totdoc,kernel_parm);
  r_delta_sq=r_delta*r_delta;

  r_delta_avg=estimate_r_delta_average(docs,totdoc,kernel_parm);
  if(learn_parm->svm_c == 0.0) {  /* default value for C */
    learn_parm->svm_c=1.0/(r_delta_avg*r_delta_avg);
    if(verbosity>=1) 
      printf("Setting default regularization parameter C=%.4f\n",
	     learn_parm->svm_c);
  }

  learn_parm->biased_hyperplane=0; /* learn an unbiased hyperplane */

  learn_parm->eps=0.0;      /* No margin, unless explicitly handcoded
                               in the right-hand side in the training
                               set.  */

  for(i=0;i<totdoc;i++) {    /* various inits */
    docs[i]->docnum=i;
    a[i]=0;
    lin[i]=0;
    c[i]=rhs[i];       /* set right-hand side */
    unlabeled[i]=0;
    inconsistent[i]=0;
    learn_parm->svm_cost[i]=learn_parm->svm_c*learn_parm->svm_costratio*
      docs[i]->costfactor;
    label[i]=1;
  }
  if(learn_parm->sharedslack) /* if shared slacks are used, they must */
    for(i=0;i<totdoc;i++)     /*  be used on every constraint */
      if(!docs[i]->slackid) {
	perror("Error: Missing shared slacks definitions in some of the examples.");
	exit(0);
      }
      
  /* compute starting state for initial alpha values */
  if(alpha) {
    if(verbosity>=1) {
      printf("Computing starting state..."); fflush(stdout);
    }
    index = (long *)my_malloc(sizeof(long)*totdoc);
    index2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
    weights=(double *)my_malloc(sizeof(double)*(totwords+1));
    aicache = (CFLOAT *)my_malloc(sizeof(CFLOAT)*totdoc);
    for(i=0;i<totdoc;i++) {    /* create full index and clip alphas */
      index[i]=1;
      alpha[i]=fabs(alpha[i]);
      if(alpha[i]<0) alpha[i]=0;
      if(alpha[i]>learn_parm->svm_cost[i]) alpha[i]=learn_parm->svm_cost[i];
    }
    if(kernel_parm->kernel_type != LINEAR) {
      for(i=0;i<totdoc;i++)     /* fill kernel cache with unbounded SV */
	if((alpha[i]>0) && (alpha[i]<learn_parm->svm_cost[i]) 
	   && (kernel_cache_space_available(kernel_cache))) 
	  cache_kernel_row(kernel_cache,docs,i,kernel_parm);
      for(i=0;i<totdoc;i++)     /* fill rest of kernel cache with bounded SV */
	if((alpha[i]==learn_parm->svm_cost[i]) 
	   && (kernel_cache_space_available(kernel_cache))) 
	  cache_kernel_row(kernel_cache,docs,i,kernel_parm);
    }
    (void)compute_index(index,totdoc,index2dnum);
    update_linear_component(docs,label,index2dnum,alpha,a,index2dnum,totdoc,
			    totwords,kernel_parm,kernel_cache,lin,aicache,
			    weights);
    (void)calculate_svm_model(docs,label,unlabeled,lin,alpha,a,c,
			      learn_parm,index2dnum,index2dnum,model);
    for(i=0;i<totdoc;i++) {    /* copy initial alphas */
      a[i]=alpha[i];
    }
    free(index);
    free(index2dnum);
    free(weights);
    free(aicache);
    if(verbosity>=1) {
      printf("done.\n");  fflush(stdout);
    }   
  } 

  /* removing inconsistent does not work for general optimization problem */
  if(learn_parm->remove_inconsistent) {	  
    learn_parm->remove_inconsistent = 0;
    printf("'remove inconsistent' not available in this mode. Switching option off!"); fflush(stdout);
  }

  /* caching makes no sense for linear kernel */
  if(kernel_parm->kernel_type == LINEAR) {
    kernel_cache = NULL;   
  } 

  if(verbosity==1) {
    printf("Optimizing"); fflush(stdout);
  }

  /* train the svm */
  if(learn_parm->sharedslack)
    iterations=optimize_to_convergence_sharedslack(docs,label,totdoc,
				     totwords,learn_parm,kernel_parm,
				     kernel_cache,&shrink_state,model,
				     a,lin,c,&timing_profile,
				     &maxdiff);
  else
    iterations=optimize_to_convergence(docs,label,totdoc,
				     totwords,learn_parm,kernel_parm,
				     kernel_cache,&shrink_state,model,
				     inconsistent,unlabeled,
				     a,lin,c,&timing_profile,
				     &maxdiff,(long)-1,(long)1);
  
  if(verbosity>=1) {
    if(verbosity==1) printf("done. (%ld iterations)\n",iterations);

    misclassified=0;
    for(i=0;(i<totdoc);i++) { /* get final statistic */
      if((lin[i]-model->b)*(double)label[i] <= 0.0) 
	misclassified++;
    }

    printf("Optimization finished (maxdiff=%.5f).\n",maxdiff); 

    runtime_end=get_runtime();
    if(verbosity>=2) {
      printf("Runtime in cpu-seconds: %.2f (%.2f%% for kernel/%.2f%% for optimizer/%.2f%% for final/%.2f%% for update/%.2f%% for model/%.2f%% for check/%.2f%% for select)\n",
        ((float)runtime_end-(float)runtime_start)/100.0,
        (100.0*timing_profile.time_kernel)/(float)(runtime_end-runtime_start),
	(100.0*timing_profile.time_opti)/(float)(runtime_end-runtime_start),
	(100.0*timing_profile.time_shrink)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_update)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_model)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_check)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_select)/(float)(runtime_end-runtime_start));
    }
    else {
      printf("Runtime in cpu-seconds: %.2f\n",
	     (runtime_end-runtime_start)/100.0);
    }
  }
  if((verbosity>=1) && (!learn_parm->skip_final_opt_check)) {
    loss=0;
    model_length=0; 
    for(i=0;i<totdoc;i++) {
      if((lin[i]-model->b)*(double)label[i] < c[i]-learn_parm->epsilon_crit)
	loss+=c[i]-(lin[i]-model->b)*(double)label[i];
      model_length+=a[i]*label[i]*lin[i];
    }
    model_length=sqrt(model_length);
    fprintf(stdout,"Norm of weight vector: |w|=%.5f\n",model_length);
  }
  
  if(learn_parm->sharedslack) {
    index = (long *)my_malloc(sizeof(long)*totdoc);
    index2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
    maxslackid=0;
    for(i=0;i<totdoc;i++) {    /* create full index */
      index[i]=1;
      if(maxslackid<docs[i]->slackid)
	maxslackid=docs[i]->slackid;
    }
    (void)compute_index(index,totdoc,index2dnum);
    slack=(double *)my_malloc(sizeof(double)*(maxslackid+1));
    alphaslack=(double *)my_malloc(sizeof(double)*(maxslackid+1));
    for(i=0;i<=maxslackid;i++) {    /* init shared slacks */
      slack[i]=0;
      alphaslack[i]=0;
    }
    compute_shared_slacks(docs,label,a,lin,c,index2dnum,learn_parm,
			  slack,alphaslack);
    loss=0;
    model->at_upper_bound=0;
    svsetnum=0;
    for(i=0;i<=maxslackid;i++) {    /* create full index */
      loss+=slack[i];
      if(alphaslack[i] > (learn_parm->svm_c - learn_parm->epsilon_a)) 
	model->at_upper_bound++;
      if(alphaslack[i] > learn_parm->epsilon_a)
	svsetnum++;
    }
    free(index);
    free(index2dnum);
    free(slack);
    free(alphaslack);
  }
  
  if((verbosity>=1) && (!learn_parm->skip_final_opt_check)) {
    if(learn_parm->sharedslack) {
      printf("Number of SV: %ld\n",
	     model->sv_num-1);
      printf("Number of non-zero slack variables: %ld (out of %ld)\n",
	     model->at_upper_bound,svsetnum);
      fprintf(stdout,"L1 loss: loss=%.5f\n",loss);
    }
    else {
      upsupvecnum=0;
      for(i=1;i<model->sv_num;i++) {
	if(fabs(model->alpha[i]) >= 
	   (learn_parm->svm_cost[(model->supvec[i])->docnum]-
	    learn_parm->epsilon_a)) 
	  upsupvecnum++;
      }
      printf("Number of SV: %ld (including %ld at upper bound)\n",
	     model->sv_num-1,upsupvecnum);
      fprintf(stdout,"L1 loss: loss=%.5f\n",loss);
    }
    example_length=estimate_sphere(model,kernel_parm); 
    fprintf(stdout,"Norm of longest example vector: |x|=%.5f\n",
	    length_of_longest_document_vector(docs,totdoc,kernel_parm));
  }
  if(verbosity>=1) {
    printf("Number of kernel evaluations: %ld\n",kernel_cache_statistic);
  }
    
  if(alpha) {
    for(i=0;i<totdoc;i++) {    /* copy final alphas */
      alpha[i]=a[i];
    }
  }
 
  if(learn_parm->alphafile[0])
    write_alphas(learn_parm->alphafile,a,label,totdoc);
  
  shrink_state_cleanup(&shrink_state);
  free(label);
  free(unlabeled);
  free(inconsistent);
  free(c);
  free(a);
  free(lin);
  free(learn_parm->svm_cost);
}


long optimize_to_convergence(DOC **docs, long int *label, long int totdoc, 
			     long int totwords, LEARN_PARM *learn_parm, 
			     KERNEL_PARM *kernel_parm, 
			     KERNEL_CACHE *kernel_cache, 
			     SHRINK_STATE *shrink_state, MODEL *model, 
			     long int *inconsistent, long int *unlabeled, 
			     double *a, double *lin, double *c, 
			     TIMING *timing_profile, double *maxdiff, 
			     long int heldout, long int retrain)
     /* docs: Training vectors (x-part) */
     /* label: Training labels/value (y-part, zero if test example for
			      transduction) */
     /* totdoc: Number of examples in docs/label */
     /* totwords: Number of features (i.e. highest feature index) */
     /* laern_parm: Learning paramenters */
     /* kernel_parm: Kernel paramenters */
     /* kernel_cache: Initialized/partly filled Cache, if using a kernel. 
                      NULL if linear. */
     /* shrink_state: State of active variables */
     /* model: Returns learning result */
     /* inconsistent: examples thrown out as inconstistent */
     /* unlabeled: test examples for transduction */
     /* a: alphas */
     /* lin: linear component of gradient */
     /* c: right hand side of inequalities (margin) */
     /* maxdiff: returns maximum violation of KT-conditions */
     /* heldout: marks held-out example for leave-one-out (or -1) */
     /* retrain: selects training mode (1=regular / 2=holdout) */
{
  long *chosen,*key,i,j,jj,*last_suboptimal_at,noshrink;
  long inconsistentnum,choosenum,already_chosen=0,iteration;
  long misclassified,supvecnum=0,*active2dnum,inactivenum;
  long *working2dnum,*selexam;
  long activenum;
  double criterion,eq;
  double *a_old;
  long t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0; /* timing */
  long transductcycle;
  long transduction;
  double epsilon_crit_org; 
  double bestmaxdiff;
  long   bestmaxdiffiter,terminate;

  double *selcrit;  /* buffer for sorting */        
  CFLOAT *aicache;  /* buffer to keep one row of hessian */
  double *weights;  /* buffer for weight vector in linear case */
  QP qp;            /* buffer for one quadratic program */

  epsilon_crit_org=learn_parm->epsilon_crit; /* save org */
  if(kernel_parm->kernel_type == LINEAR) {
    learn_parm->epsilon_crit=2.0;
    kernel_cache=NULL;   /* caching makes no sense for linear kernel */
  } 
  learn_parm->epsilon_shrink=2;
  (*maxdiff)=1;

  learn_parm->totwords=totwords;

  chosen = (long *)my_malloc(sizeof(long)*totdoc);
  last_suboptimal_at = (long *)my_malloc(sizeof(long)*totdoc);
  key = (long *)my_malloc(sizeof(long)*(totdoc+11)); 
  selcrit = (double *)my_malloc(sizeof(double)*totdoc);
  selexam = (long *)my_malloc(sizeof(long)*totdoc);
  a_old = (double *)my_malloc(sizeof(double)*totdoc);
  aicache = (CFLOAT *)my_malloc(sizeof(CFLOAT)*totdoc);
  working2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
  active2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
  qp.opt_ce = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_ce0 = (double *)my_malloc(sizeof(double));
  qp.opt_g = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize
				 *learn_parm->svm_maxqpsize);
  qp.opt_g0 = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_xinit = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_low=(double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_up=(double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  weights=(double *)my_malloc(sizeof(double)*(totwords+1));

  choosenum=0;
  inconsistentnum=0;
  transductcycle=0;
  transduction=0;
  if(!retrain) retrain=1;
  iteration=1;
  bestmaxdiffiter=1;
  bestmaxdiff=999999999;
  terminate=0;

  if(kernel_cache) {
    kernel_cache->time=iteration;  /* for lru cache */
    kernel_cache_reset_lru(kernel_cache);
  }

  for(i=0;i<totdoc;i++) {    /* various inits */
    chosen[i]=0;
    a_old[i]=a[i];
    last_suboptimal_at[i]=1;
    if(inconsistent[i]) 
      inconsistentnum++;
    if(unlabeled[i]) {
      transduction=1;
    }
  }
  activenum=compute_index(shrink_state->active,totdoc,active2dnum);
  inactivenum=totdoc-activenum;
  clear_index(working2dnum);

                            /* repeat this loop until we have convergence */
  for(;retrain && (!terminate);iteration++) {

    if(kernel_cache)
      kernel_cache->time=iteration;  /* for lru cache */
    if(verbosity>=2) {
      printf(
	"Iteration %ld: ",iteration); fflush(stdout);
    }
    else if(verbosity==1) {
      printf("."); fflush(stdout);
    }

    if(verbosity>=2) t0=get_runtime();
    if(verbosity>=3) {
      printf("\nSelecting working set... "); fflush(stdout); 
    }

    if(learn_parm->svm_newvarsinqp>learn_parm->svm_maxqpsize) 
      learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize;

    i=0;
    for(jj=0;(j=working2dnum[jj])>=0;jj++) { /* clear working set */
      if((chosen[j]>=(learn_parm->svm_maxqpsize/
		      minl(learn_parm->svm_maxqpsize,
			   learn_parm->svm_newvarsinqp))) 
	 || (inconsistent[j])
	 || (j == heldout)) {
	chosen[j]=0; 
	choosenum--; 
      }
      else {
	chosen[j]++;
	working2dnum[i++]=j;
      }
    }
    working2dnum[i]=-1;

    if(retrain == 2) {
      choosenum=0;
      for(jj=0;(j=working2dnum[jj])>=0;jj++) { /* fully clear working set */
	chosen[j]=0; 
      }
      clear_index(working2dnum);
      for(i=0;i<totdoc;i++) { /* set inconsistent examples to zero (-i 1) */
	if((inconsistent[i] || (heldout==i)) && (a[i] != 0.0)) {
	  chosen[i]=99999;
	  choosenum++;
	  a[i]=0;
	}
      }
      if(learn_parm->biased_hyperplane) {
	eq=0;
	for(i=0;i<totdoc;i++) { /* make sure we fulfill equality constraint */
	  eq+=a[i]*label[i];
	}
	for(i=0;(i<totdoc) && (fabs(eq) > learn_parm->epsilon_a);i++) {
	  if((eq*label[i] > 0) && (a[i] > 0)) {
	    chosen[i]=88888;
	    choosenum++;
	    if((eq*label[i]) > a[i]) {
	      eq-=(a[i]*label[i]);
	      a[i]=0;
	    }
	    else {
	      a[i]-=(eq*label[i]);
	      eq=0;
	    }
	  }
	}
      }
      compute_index(chosen,totdoc,working2dnum);
    }
    else {      /* select working set according to steepest gradient */
      if(iteration % 101) {
        already_chosen=0;
	if((minl(learn_parm->svm_newvarsinqp,
		 learn_parm->svm_maxqpsize-choosenum)>=4) 
	   && (kernel_parm->kernel_type != LINEAR)) {
	  /* select part of the working set from cache */
	  already_chosen=select_next_qp_subproblem_grad(
			      label,unlabeled,a,lin,c,totdoc,
			      (long)(minl(learn_parm->svm_maxqpsize-choosenum,
					  learn_parm->svm_newvarsinqp)
				     /2),
			      learn_parm,inconsistent,active2dnum,
			      working2dnum,selcrit,selexam,kernel_cache,1,
			      key,chosen);
	  choosenum+=already_chosen;
	}
	choosenum+=select_next_qp_subproblem_grad(
                              label,unlabeled,a,lin,c,totdoc,
                              minl(learn_parm->svm_maxqpsize-choosenum,
				   learn_parm->svm_newvarsinqp-already_chosen),
                              learn_parm,inconsistent,active2dnum,
			      working2dnum,selcrit,selexam,kernel_cache,0,key,
			      chosen);
      }
      else { /* once in a while, select a somewhat random working set
		to get unlocked of infinite loops due to numerical
		inaccuracies in the core qp-solver */
	choosenum+=select_next_qp_subproblem_rand(
                              label,unlabeled,a,lin,c,totdoc,
                              minl(learn_parm->svm_maxqpsize-choosenum,
				   learn_parm->svm_newvarsinqp),
                              learn_parm,inconsistent,active2dnum,
			      working2dnum,selcrit,selexam,kernel_cache,key,
			      chosen,iteration);
      }
    }

    if(verbosity>=2) {
      printf(" %ld vectors chosen\n",choosenum); fflush(stdout); 
    }

    if(verbosity>=2) t1=get_runtime();

    if(kernel_cache) 
      cache_multiple_kernel_rows(kernel_cache,docs,working2dnum,
				 choosenum,kernel_parm); 
    
    if(verbosity>=2) t2=get_runtime();
    if(retrain != 2) {
      optimize_svm(docs,label,unlabeled,inconsistent,0.0,chosen,active2dnum,
		   model,totdoc,working2dnum,choosenum,a,lin,c,learn_parm,
		   aicache,kernel_parm,&qp,&epsilon_crit_org);
    }

    if(verbosity>=2) t3=get_runtime();
    update_linear_component(docs,label,active2dnum,a,a_old,working2dnum,totdoc,
			    totwords,kernel_parm,kernel_cache,lin,aicache,
			    weights);

    if(verbosity>=2) t4=get_runtime();
    supvecnum=calculate_svm_model(docs,label,unlabeled,lin,a,a_old,c,
		                  learn_parm,working2dnum,active2dnum,model);

    if(verbosity>=2) t5=get_runtime();

    /* The following computation of the objective function works only */
    /* relative to the active variables */
    if(verbosity>=3) {
      criterion=compute_objective_function(a,lin,c,learn_parm->eps,label,
		                           active2dnum);
      printf("Objective function (over active variables): %.16f\n",criterion);
      fflush(stdout); 
    }

    for(jj=0;(i=working2dnum[jj])>=0;jj++) {
      a_old[i]=a[i];
    }

    if(retrain == 2) {  /* reset inconsistent unlabeled examples */
      for(i=0;(i<totdoc);i++) {
	if(inconsistent[i] && unlabeled[i]) {
	  inconsistent[i]=0;
	  label[i]=0;
	}
      }
    }

    retrain=check_optimality(model,label,unlabeled,a,lin,c,totdoc,learn_parm,
			     maxdiff,epsilon_crit_org,&misclassified,
			     inconsistent,active2dnum,last_suboptimal_at,
			     iteration,kernel_parm);

    if(verbosity>=2) {
      t6=get_runtime();
      timing_profile->time_select+=t1-t0;
      timing_profile->time_kernel+=t2-t1;
      timing_profile->time_opti+=t3-t2;
      timing_profile->time_update+=t4-t3;
      timing_profile->time_model+=t5-t4;
      timing_profile->time_check+=t6-t5;
    }

    /* checking whether optimizer got stuck */
    if((*maxdiff) < bestmaxdiff) {
      bestmaxdiff=(*maxdiff);
      bestmaxdiffiter=iteration;
    }
    if(iteration > (bestmaxdiffiter+learn_parm->maxiter)) { 
      /* long time no progress? */
      terminate=1;
      retrain=0;
      if(verbosity>=1) 
	printf("\nWARNING: Relaxing KT-Conditions due to slow progress! Terminating!\n");
    }

    noshrink=0;
    if((!retrain) && (inactivenum>0) 
       && ((!learn_parm->skip_final_opt_check) 
	   || (kernel_parm->kernel_type == LINEAR))) { 
      if(((verbosity>=1) && (kernel_parm->kernel_type != LINEAR)) 
	 || (verbosity>=2)) {
	if(verbosity==1) {
	  printf("\n");
	}
	printf(" Checking optimality of inactive variables..."); 
	fflush(stdout);
      }
      t1=get_runtime();
      reactivate_inactive_examples(label,unlabeled,a,shrink_state,lin,c,totdoc,
				   totwords,iteration,learn_parm,inconsistent,
				   docs,kernel_parm,kernel_cache,model,aicache,
				   weights,maxdiff);
      /* Update to new active variables. */
      activenum=compute_index(shrink_state->active,totdoc,active2dnum);
      inactivenum=totdoc-activenum;
      /* reset watchdog */
      bestmaxdiff=(*maxdiff);
      bestmaxdiffiter=iteration;
      /* termination criterion */
      noshrink=1;
      retrain=0;
      if((*maxdiff) > learn_parm->epsilon_crit) 
	retrain=1;
      timing_profile->time_shrink+=get_runtime()-t1;
      if(((verbosity>=1) && (kernel_parm->kernel_type != LINEAR)) 
	 || (verbosity>=2)) {
	printf("done.\n");  fflush(stdout);
        printf(" Number of inactive variables = %ld\n",inactivenum);
      }		  
    }

    if((!retrain) && (learn_parm->epsilon_crit>(*maxdiff))) 
      learn_parm->epsilon_crit=(*maxdiff);
    if((!retrain) && (learn_parm->epsilon_crit>epsilon_crit_org)) {
      learn_parm->epsilon_crit/=2.0;
      retrain=1;
      noshrink=1;
    }
    if(learn_parm->epsilon_crit<epsilon_crit_org) 
      learn_parm->epsilon_crit=epsilon_crit_org;
    
    if(verbosity>=2) {
      printf(" => (%ld SV (incl. %ld SV at u-bound), max violation=%.5f)\n",
	     supvecnum,model->at_upper_bound,(*maxdiff)); 
      fflush(stdout);
    }
    if(verbosity>=3) {
      printf("\n");
    }

    if((!retrain) && (transduction)) {
      for(i=0;(i<totdoc);i++) {
	shrink_state->active[i]=1;
      }
      activenum=compute_index(shrink_state->active,totdoc,active2dnum);
      inactivenum=0;
      if(verbosity==1) printf("done\n");
      retrain=incorporate_unlabeled_examples(model,label,inconsistent,
					     unlabeled,a,lin,totdoc,
					     selcrit,selexam,key,
					     transductcycle,kernel_parm,
					     learn_parm);
      epsilon_crit_org=learn_parm->epsilon_crit;
      if(kernel_parm->kernel_type == LINEAR)
	learn_parm->epsilon_crit=1; 
      transductcycle++;
      /* reset watchdog */
      bestmaxdiff=(*maxdiff);
      bestmaxdiffiter=iteration;
    } 
    else if(((iteration % 10) == 0) && (!noshrink)) {
      activenum=shrink_problem(docs,learn_parm,shrink_state,kernel_parm,
			       active2dnum,last_suboptimal_at,iteration,totdoc,
			       maxl((long)(activenum/10),
				    maxl((long)(totdoc/500),100)),
			       a,inconsistent);
      inactivenum=totdoc-activenum;
      if((kernel_cache)
	 && (supvecnum>kernel_cache->max_elems)
	 && ((kernel_cache->activenum-activenum)>maxl((long)(activenum/10),500))) {
	kernel_cache_shrink(kernel_cache,totdoc,
			    minl((kernel_cache->activenum-activenum),
				 (kernel_cache->activenum-supvecnum)),
			    shrink_state->active); 
      }
    }

    if((!retrain) && learn_parm->remove_inconsistent) {
      if(verbosity>=1) {
	printf(" Moving training errors to inconsistent examples...");
	fflush(stdout);
      }
      if(learn_parm->remove_inconsistent == 1) {
	retrain=identify_inconsistent(a,label,unlabeled,totdoc,learn_parm,
				      &inconsistentnum,inconsistent); 
      }
      else if(learn_parm->remove_inconsistent == 2) {
	retrain=identify_misclassified(lin,label,unlabeled,totdoc,
				       model,&inconsistentnum,inconsistent); 
      }
      else if(learn_parm->remove_inconsistent == 3) {
	retrain=identify_one_misclassified(lin,label,unlabeled,totdoc,
				   model,&inconsistentnum,inconsistent);
      }
      if(retrain) {
	if(kernel_parm->kernel_type == LINEAR) { /* reinit shrinking */
	  learn_parm->epsilon_crit=2.0;
	} 
      }
      if(verbosity>=1) {
	printf("done.\n");
	if(retrain) {
	  printf(" Now %ld inconsistent examples.\n",inconsistentnum);
	}
      }
    }
  } /* end of loop */

  free(chosen);
  free(last_suboptimal_at);
  free(key);
  free(selcrit);
  free(selexam);
  free(a_old);
  free(aicache);
  free(working2dnum);
  free(active2dnum);
  free(qp.opt_ce);
  free(qp.opt_ce0);
  free(qp.opt_g);
  free(qp.opt_g0);
  free(qp.opt_xinit);
  free(qp.opt_low);
  free(qp.opt_up);
  free(weights);

  learn_parm->epsilon_crit=epsilon_crit_org; /* restore org */
  model->maxdiff=(*maxdiff);

  return(iteration);
}

long optimize_to_convergence_sharedslack(DOC **docs, long int *label, 
			     long int totdoc, 
			     long int totwords, LEARN_PARM *learn_parm, 
			     KERNEL_PARM *kernel_parm, 
			     KERNEL_CACHE *kernel_cache, 
			     SHRINK_STATE *shrink_state, MODEL *model, 
			     double *a, double *lin, double *c, 
			     TIMING *timing_profile, double *maxdiff)
     /* docs: Training vectors (x-part) */
     /* label: Training labels/value (y-part, zero if test example for
			      transduction) */
     /* totdoc: Number of examples in docs/label */
     /* totwords: Number of features (i.e. highest feature index) */
     /* learn_parm: Learning paramenters */
     /* kernel_parm: Kernel paramenters */
     /* kernel_cache: Initialized/partly filled Cache, if using a kernel. 
                      NULL if linear. */
     /* shrink_state: State of active variables */
     /* model: Returns learning result */
     /* a: alphas */
     /* lin: linear component of gradient */
     /* c: right hand side of inequalities (margin) */
     /* maxdiff: returns maximum violation of KT-conditions */
{
  long *chosen,*key,i,j,jj,*last_suboptimal_at,noshrink,*unlabeled;
  long *inconsistent,choosenum,already_chosen=0,iteration;
  long misclassified,supvecnum=0,*active2dnum,inactivenum;
  long *working2dnum,*selexam,*ignore;
  long activenum,retrain,maxslackid,slackset,jointstep;
  double criterion,eq_target;
  double *a_old,*alphaslack;
  long t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0; /* timing */
  double epsilon_crit_org,maxsharedviol; 
  double bestmaxdiff;
  long   bestmaxdiffiter,terminate;

  double *selcrit;  /* buffer for sorting */        
  CFLOAT *aicache;  /* buffer to keep one row of hessian */
  double *weights;  /* buffer for weight vector in linear case */
  QP qp;            /* buffer for one quadratic program */
  double *slack;    /* vector of slack variables for optimization with
		       shared slacks */

  epsilon_crit_org=learn_parm->epsilon_crit; /* save org */
  if(kernel_parm->kernel_type == LINEAR) {
    learn_parm->epsilon_crit=2.0;
    kernel_cache=NULL;   /* caching makes no sense for linear kernel */
  } 
  learn_parm->epsilon_shrink=2;
  (*maxdiff)=1;

  learn_parm->totwords=totwords;

  chosen = (long *)my_malloc(sizeof(long)*totdoc);
  unlabeled = (long *)my_malloc(sizeof(long)*totdoc);
  inconsistent = (long *)my_malloc(sizeof(long)*totdoc);
  ignore = (long *)my_malloc(sizeof(long)*totdoc);
  key = (long *)my_malloc(sizeof(long)*(totdoc+11)); 
  selcrit = (double *)my_malloc(sizeof(double)*totdoc);
  selexam = (long *)my_malloc(sizeof(long)*totdoc);
  a_old = (double *)my_malloc(sizeof(double)*totdoc);
  aicache = (CFLOAT *)my_malloc(sizeof(CFLOAT)*totdoc);
  working2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
  active2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
  qp.opt_ce = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_ce0 = (double *)my_malloc(sizeof(double));
  qp.opt_g = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize
				 *learn_parm->svm_maxqpsize);
  qp.opt_g0 = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_xinit = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_low=(double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_up=(double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  weights=(double *)my_malloc(sizeof(double)*(totwords+1));
  maxslackid=0;
  for(i=0;i<totdoc;i++) {    /* determine size of slack array */
    if(maxslackid<docs[i]->slackid)
      maxslackid=docs[i]->slackid;
  }
  slack=(double *)my_malloc(sizeof(double)*(maxslackid+1));
  alphaslack=(double *)my_malloc(sizeof(double)*(maxslackid+1));
  last_suboptimal_at = (long *)my_malloc(sizeof(long)*(maxslackid+1));
  for(i=0;i<=maxslackid;i++) {    /* init shared slacks */
    slack[i]=0;
    alphaslack[i]=0;
    last_suboptimal_at[i]=1;
  }

  choosenum=0;
  retrain=1;
  iteration=1;
  bestmaxdiffiter=1;
  bestmaxdiff=999999999;
  terminate=0;

  if(kernel_cache) {
    kernel_cache->time=iteration;  /* for lru cache */
    kernel_cache_reset_lru(kernel_cache);
  }

  for(i=0;i<totdoc;i++) {    /* various inits */
    chosen[i]=0;
    unlabeled[i]=0;
    inconsistent[i]=0;
    ignore[i]=0;
    a_old[i]=a[i];
  }
  activenum=compute_index(shrink_state->active,totdoc,active2dnum);
  inactivenum=totdoc-activenum;
  clear_index(working2dnum);

  /* call to init slack and alphaslack */
  compute_shared_slacks(docs,label,a,lin,c,active2dnum,learn_parm,
			slack,alphaslack);

                            /* repeat this loop until we have convergence */
  for(;retrain && (!terminate);iteration++) {

    if(kernel_cache)
      kernel_cache->time=iteration;  /* for lru cache */
    if(verbosity>=2) {
      printf(
	"Iteration %ld: ",iteration); fflush(stdout);
    }
    else if(verbosity==1) {
      printf("."); fflush(stdout);
    }

    if(verbosity>=2) t0=get_runtime();
    if(verbosity>=3) {
      printf("\nSelecting working set... "); fflush(stdout); 
    }

    if(learn_parm->svm_newvarsinqp>learn_parm->svm_maxqpsize) 
      learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize;

    /* select working set according to steepest gradient */
    jointstep=0;
    eq_target=0;
    if(iteration % 101) {
      slackset=select_next_qp_slackset(docs,label,a,lin,slack,alphaslack,c,
				       learn_parm,active2dnum,&maxsharedviol);
      if((iteration % 2) 
	 || (!slackset) || (maxsharedviol<learn_parm->epsilon_crit)){
	/* do a step with examples from different slack sets */
	if(verbosity >= 2) {
	  printf("(i-step)"); fflush(stdout);
	}
	i=0;
	for(jj=0;(j=working2dnum[jj])>=0;jj++) { /* clear old part of working set */
	  if((chosen[j]>=(learn_parm->svm_maxqpsize/
			  minl(learn_parm->svm_maxqpsize,
			       learn_parm->svm_newvarsinqp)))) {
	    chosen[j]=0; 
	    choosenum--; 
	  }
	  else {
	    chosen[j]++;
	    working2dnum[i++]=j;
	  }
	}
	working2dnum[i]=-1;
	
	already_chosen=0;
	if((minl(learn_parm->svm_newvarsinqp,
		 learn_parm->svm_maxqpsize-choosenum)>=4) 
	   && (kernel_parm->kernel_type != LINEAR)) {
	  /* select part of the working set from cache */
	  already_chosen=select_next_qp_subproblem_grad(
			      label,unlabeled,a,lin,c,totdoc,
			      (long)(minl(learn_parm->svm_maxqpsize-choosenum,
					  learn_parm->svm_newvarsinqp)
				     /2),
			      learn_parm,inconsistent,active2dnum,
			      working2dnum,selcrit,selexam,kernel_cache,
			      (long)1,key,chosen);
	  choosenum+=already_chosen;
	}
	choosenum+=select_next_qp_subproblem_grad(
                              label,unlabeled,a,lin,c,totdoc,
                              minl(learn_parm->svm_maxqpsize-choosenum,
				   learn_parm->svm_newvarsinqp-already_chosen),
                              learn_parm,inconsistent,active2dnum,
			      working2dnum,selcrit,selexam,kernel_cache,
			      (long)0,key,chosen);
      }
      else { /* do a step with all examples from same slack set */
	if(verbosity >= 2) {
	  printf("(j-step on %ld)",slackset); fflush(stdout);
	}
	jointstep=1;
	for(jj=0;(j=working2dnum[jj])>=0;jj++) { /* clear working set */
	    chosen[j]=0; 
	}
	working2dnum[0]=-1;
	eq_target=alphaslack[slackset];
	for(j=0;j<totdoc;j++) {                  /* mask all but slackset */
	  /* for(jj=0;(j=active2dnum[jj])>=0;jj++) { */
	  if(docs[j]->slackid != slackset)
	    ignore[j]=1; 
	  else {
	    ignore[j]=0; 
	    learn_parm->svm_cost[j]=learn_parm->svm_c;
	    /* printf("Inslackset(%ld,%ld)",j,shrink_state->active[j]); */
	  }
	}
	learn_parm->biased_hyperplane=1;
	choosenum=select_next_qp_subproblem_grad(
                              label,unlabeled,a,lin,c,totdoc,
                              learn_parm->svm_maxqpsize,
                              learn_parm,ignore,active2dnum,
			      working2dnum,selcrit,selexam,kernel_cache,
			      (long)0,key,chosen);
	learn_parm->biased_hyperplane=0;
      }
    }
    else { /* once in a while, select a somewhat random working set
	      to get unlocked of infinite loops due to numerical
	      inaccuracies in the core qp-solver */
      choosenum+=select_next_qp_subproblem_rand(
                              label,unlabeled,a,lin,c,totdoc,
                              minl(learn_parm->svm_maxqpsize-choosenum,
				   learn_parm->svm_newvarsinqp),
                              learn_parm,inconsistent,active2dnum,
			      working2dnum,selcrit,selexam,kernel_cache,key,
			      chosen,iteration);
    }

    if(verbosity>=2) {
      printf(" %ld vectors chosen\n",choosenum); fflush(stdout); 
    }

    if(verbosity>=2) t1=get_runtime();

    if(kernel_cache) 
      cache_multiple_kernel_rows(kernel_cache,docs,working2dnum,
				 choosenum,kernel_parm); 

    if(verbosity>=2) t2=get_runtime();
    if(jointstep) learn_parm->biased_hyperplane=1;
    optimize_svm(docs,label,unlabeled,ignore,eq_target,chosen,active2dnum,
		 model,totdoc,working2dnum,choosenum,a,lin,c,learn_parm,
		 aicache,kernel_parm,&qp,&epsilon_crit_org);
    learn_parm->biased_hyperplane=0;

    for(jj=0;(i=working2dnum[jj])>=0;jj++)   /* recompute sums of alphas */
      alphaslack[docs[i]->slackid]+=(a[i]-a_old[i]);
    for(jj=0;(i=working2dnum[jj])>=0;jj++) { /* reduce alpha to fulfill
						constraints */
      if(alphaslack[docs[i]->slackid] > learn_parm->svm_c) {
	if(a[i] < (alphaslack[docs[i]->slackid]-learn_parm->svm_c)) {
	  alphaslack[docs[i]->slackid]-=a[i];
	  a[i]=0;
	}
	else {
	  a[i]-=(alphaslack[docs[i]->slackid]-learn_parm->svm_c);
	  alphaslack[docs[i]->slackid]=learn_parm->svm_c;
	}
      }
    }
    for(jj=0;(i=active2dnum[jj])>=0;jj++) 
      learn_parm->svm_cost[i]=a[i]+(learn_parm->svm_c
				    -alphaslack[docs[i]->slackid]);

    if(verbosity>=2) t3=get_runtime();
    update_linear_component(docs,label,active2dnum,a,a_old,working2dnum,totdoc,
			    totwords,kernel_parm,kernel_cache,lin,aicache,
			    weights);
    compute_shared_slacks(docs,label,a,lin,c,active2dnum,learn_parm,
			  slack,alphaslack);

    if(verbosity>=2) t4=get_runtime();
    supvecnum=calculate_svm_model(docs,label,unlabeled,lin,a,a_old,c,
		                  learn_parm,working2dnum,active2dnum,model);

    if(verbosity>=2) t5=get_runtime();

    /* The following computation of the objective function works only */
    /* relative to the active variables */
    if(verbosity>=3) {
      criterion=compute_objective_function(a,lin,c,learn_parm->eps,label,
		                           active2dnum);
      printf("Objective function (over active variables): %.16f\n",criterion);
      fflush(stdout); 
    }

    for(jj=0;(i=working2dnum[jj])>=0;jj++) {
      a_old[i]=a[i];
    }

    retrain=check_optimality_sharedslack(docs,model,label,a,lin,c,
                             slack,alphaslack,totdoc,learn_parm,
			     maxdiff,epsilon_crit_org,&misclassified,
			     active2dnum,last_suboptimal_at,
			     iteration,kernel_parm);

    if(verbosity>=2) {
      t6=get_runtime();
      timing_profile->time_select+=t1-t0;
      timing_profile->time_kernel+=t2-t1;
      timing_profile->time_opti+=t3-t2;
      timing_profile->time_update+=t4-t3;
      timing_profile->time_model+=t5-t4;
      timing_profile->time_check+=t6-t5;
    }

    /* checking whether optimizer got stuck */
    if((*maxdiff) < bestmaxdiff) {
      bestmaxdiff=(*maxdiff);
      bestmaxdiffiter=iteration;
    }
    if(iteration > (bestmaxdiffiter+learn_parm->maxiter)) { 
      /* long time no progress? */
      terminate=1;
      retrain=0;
      if(verbosity>=1) 
	printf("\nWARNING: Relaxing KT-Conditions due to slow progress! Terminating!\n");
    }

    noshrink=0; 

    if((!retrain) && (inactivenum>0) 
       && ((!learn_parm->skip_final_opt_check) 
	   || (kernel_parm->kernel_type == LINEAR))) { 
      if(((verbosity>=1) && (kernel_parm->kernel_type != LINEAR)) 
	 || (verbosity>=2)) {
	if(verbosity==1) {
	  printf("\n");
	}
	printf(" Checking optimality of inactive variables..."); 
	fflush(stdout);
      }
      t1=get_runtime();
      reactivate_inactive_examples(label,unlabeled,a,shrink_state,lin,c,totdoc,
				   totwords,iteration,learn_parm,inconsistent,
				   docs,kernel_parm,kernel_cache,model,aicache,
				   weights,maxdiff);
      /* Update to new active variables. */
      activenum=compute_index(shrink_state->active,totdoc,active2dnum);
      inactivenum=totdoc-activenum;
      /* check optimality, since check in reactivate does not work for
	 sharedslacks */
      retrain=check_optimality_sharedslack(docs,model,label,a,lin,c,
			     slack,alphaslack,totdoc,learn_parm,
			     maxdiff,epsilon_crit_org,&misclassified,
			     active2dnum,last_suboptimal_at,
			     iteration,kernel_parm);

      /* reset watchdog */
      bestmaxdiff=(*maxdiff);
      bestmaxdiffiter=iteration;
      /* termination criterion */
      noshrink=1;
      retrain=0;
      if((*maxdiff) > learn_parm->epsilon_crit) 
	retrain=1;
      timing_profile->time_shrink+=get_runtime()-t1;
      if(((verbosity>=1) && (kernel_parm->kernel_type != LINEAR)) 
	 || (verbosity>=2)) {
	printf("done.\n");  fflush(stdout);
        printf(" Number of inactive variables = %ld\n",inactivenum);
      }		  
    }

    if((!retrain) && (learn_parm->epsilon_crit>(*maxdiff))) 
      learn_parm->epsilon_crit=(*maxdiff);
    if((!retrain) && (learn_parm->epsilon_crit>epsilon_crit_org)) {
      learn_parm->epsilon_crit/=2.0;
      retrain=1;
      noshrink=1;
    }
    if(learn_parm->epsilon_crit<epsilon_crit_org) 
      learn_parm->epsilon_crit=epsilon_crit_org;
    
    if(verbosity>=2) {
      printf(" => (%ld SV (incl. %ld SV at u-bound), max violation=%.5f)\n",
	     supvecnum,model->at_upper_bound,(*maxdiff)); 
      fflush(stdout);
    }
    if(verbosity>=3) {
      printf("\n");
    }

    if(((iteration % 10) == 0) && (!noshrink)) {
      activenum=shrink_problem(docs,learn_parm,shrink_state,
			       kernel_parm,active2dnum,
			       last_suboptimal_at,iteration,totdoc,
			       maxl((long)(activenum/10),
				    maxl((long)(totdoc/500),100)),
			       a,inconsistent);
      inactivenum=totdoc-activenum;
      if((kernel_cache)
	 && (supvecnum>kernel_cache->max_elems)
	 && ((kernel_cache->activenum-activenum)>maxl((long)(activenum/10),500))) {
	kernel_cache_shrink(kernel_cache,totdoc,
			    minl((kernel_cache->activenum-activenum),
				 (kernel_cache->activenum-supvecnum)),
			    shrink_state->active); 
      }
    }

  } /* end of loop */


  free(alphaslack);
  free(slack);
  free(chosen);
  free(unlabeled);
  free(inconsistent);
  free(ignore);
  free(last_suboptimal_at);
  free(key);
  free(selcrit);
  free(selexam);
  free(a_old);
  free(aicache);
  free(working2dnum);
  free(active2dnum);
  free(qp.opt_ce);
  free(qp.opt_ce0);
  free(qp.opt_g);
  free(qp.opt_g0);
  free(qp.opt_xinit);
  free(qp.opt_low);
  free(qp.opt_up);
  free(weights);

  learn_parm->epsilon_crit=epsilon_crit_org; /* restore org */
  model->maxdiff=(*maxdiff);

  return(iteration);
}


double compute_objective_function(double *a, double *lin, double *c, 
				  double eps, long int *label, 
				  long int *active2dnum)
     /* Return value of objective function. */
     /* Works only relative to the active variables! */
{
  long i,ii;
  double criterion;
  /* calculate value of objective function */
  criterion=0;
  for(ii=0;active2dnum[ii]>=0;ii++) {
    i=active2dnum[ii];
    criterion=criterion+(eps-(double)label[i]*c[i])*a[i]+0.5*a[i]*label[i]*lin[i];
  } 
  return(criterion);
}

void clear_index(long int *index)
              /* initializes and empties index */
{
  index[0]=-1;
} 

void add_to_index(long int *index, long int elem)
     /* initializes and empties index */
{
  register long i;
  for(i=0;index[i] != -1;i++);
  index[i]=elem;
  index[i+1]=-1;
}

long compute_index(long int *binfeature, long int range, long int *index)
     /* create an inverted index of binfeature */
{               
  register long i,ii;

  ii=0;
  for(i=0;i<range;i++) {
    if(binfeature[i]) {
      index[ii]=i;
      ii++;
    }
  }
  for(i=0;i<4;i++) {
    index[ii+i]=-1;
  }
  return(ii);
}


void optimize_svm(DOC **docs, long int *label, long int *unlabeled, 
		  long int *exclude_from_eq_const, double eq_target,
		  long int *chosen, long int *active2dnum, MODEL *model, 
		  long int totdoc, long int *working2dnum, long int varnum, 
		  double *a, double *lin, double *c, LEARN_PARM *learn_parm, 
		  CFLOAT *aicache, KERNEL_PARM *kernel_parm, QP *qp, 
		  double *epsilon_crit_target)
     /* Do optimization on the working set. */
{
    long i;
    double *a_v;

    compute_matrices_for_optimization(docs,label,unlabeled,
				      exclude_from_eq_const,eq_target,chosen,
				      active2dnum,working2dnum,model,a,lin,c,
				      varnum,totdoc,learn_parm,aicache,
				      kernel_parm,qp);

    if(verbosity>=3) {
      printf("Running optimizer..."); fflush(stdout);
    }
    /* call the qp-subsolver */
    a_v=optimize_qp(qp,epsilon_crit_target,
		    learn_parm->svm_maxqpsize,
		    &(model->b),   /* in case the optimizer gives us */
                                   /* the threshold for free. otherwise */
                                   /* b is calculated in calculate_model. */
		    learn_parm);
    if(verbosity>=3) {         
      printf("done\n");
    }

    for(i=0;i<varnum;i++) {
      a[working2dnum[i]]=a_v[i];
      /*
      if(a_v[i]<=(0+learn_parm->epsilon_a)) {
	a[working2dnum[i]]=0;
      }
      else if(a_v[i]>=(learn_parm->svm_cost[working2dnum[i]]-learn_parm->epsilon_a)) {
	a[working2dnum[i]]=learn_parm->svm_cost[working2dnum[i]];
      }
      */
    }
}

void compute_matrices_for_optimization(DOC **docs, long int *label, 
          long int *unlabeled, long *exclude_from_eq_const, double eq_target,
	  long int *chosen, long int *active2dnum, 
          long int *key, MODEL *model, double *a, double *lin, double *c, 
	  long int varnum, long int totdoc, LEARN_PARM *learn_parm, 
          CFLOAT *aicache, KERNEL_PARM *kernel_parm, QP *qp)
{
  register long ki,kj,i,j;
  register double kernel_temp;

  if(verbosity>=3) {
    fprintf(stdout,"Computing qp-matrices (type %ld kernel [degree %ld, rbf_gamma %f, coef_lin %f, coef_const %f])...",kernel_parm->kernel_type,kernel_parm->poly_degree,kernel_parm->rbf_gamma,kernel_parm->coef_lin,kernel_parm->coef_const); 
    fflush(stdout);
  }

  qp->opt_n=varnum;
  qp->opt_ce0[0]=-eq_target; /* compute the constant for equality constraint */
  for(j=1;j<model->sv_num;j++) { /* start at 1 */
    if((!chosen[(model->supvec[j])->docnum])
       && (!exclude_from_eq_const[(model->supvec[j])->docnum])) {
      qp->opt_ce0[0]+=model->alpha[j];
    }
  } 
  if(learn_parm->biased_hyperplane) 
    qp->opt_m=1;
  else 
    qp->opt_m=0;  /* eq-constraint will be ignored */

  /* init linear part of objective function */
  for(i=0;i<varnum;i++) {
    qp->opt_g0[i]=lin[key[i]];
  }

  for(i=0;i<varnum;i++) {
    ki=key[i];

    /* Compute the matrix for equality constraints */
    qp->opt_ce[i]=label[ki];
    qp->opt_low[i]=0;
    qp->opt_up[i]=learn_parm->svm_cost[ki];

    kernel_temp=(double)kernel(kernel_parm,docs[ki],docs[ki]); 
    /* compute linear part of objective function */
    qp->opt_g0[i]-=(kernel_temp*a[ki]*(double)label[ki]); 
    /* compute quadratic part of objective function */
    qp->opt_g[varnum*i+i]=kernel_temp;
    for(j=i+1;j<varnum;j++) {
      kj=key[j];
      kernel_temp=(double)kernel(kernel_parm,docs[ki],docs[kj]);
      /* compute linear part of objective function */
      qp->opt_g0[i]-=(kernel_temp*a[kj]*(double)label[kj]);
      qp->opt_g0[j]-=(kernel_temp*a[ki]*(double)label[ki]); 
      /* compute quadratic part of objective function */
      qp->opt_g[varnum*i+j]=(double)label[ki]*(double)label[kj]*kernel_temp;
      qp->opt_g[varnum*j+i]=(double)label[ki]*(double)label[kj]*kernel_temp;
    }

    if(verbosity>=3) {
      if(i % 20 == 0) {
	fprintf(stdout,"%ld..",i); fflush(stdout);
      }
    }
  }

  for(i=0;i<varnum;i++) {
    /* assure starting at feasible point */
    qp->opt_xinit[i]=a[key[i]];
    /* set linear part of objective function */
    qp->opt_g0[i]=(learn_parm->eps-(double)label[key[i]]*c[key[i]])+qp->opt_g0[i]*(double)label[key[i]];    
  }

  if(verbosity>=3) {
    fprintf(stdout,"done\n");
  }
}

long calculate_svm_model(DOC **docs, long int *label, long int *unlabeled, 
			 double *lin, double *a, double *a_old, double *c, 
			 LEARN_PARM *learn_parm, long int *working2dnum, 
			 long int *active2dnum, MODEL *model)
     /* Compute decision function based on current values */
     /* of alpha. */
{
  long i,ii,pos,b_calculated=0,first_low,first_high;
  double ex_c,b_temp,b_low,b_high;

  if(verbosity>=3) {
    printf("Calculating model..."); fflush(stdout);
  }

  if(!learn_parm->biased_hyperplane) {
    model->b=0;
    b_calculated=1;
  }

  for(ii=0;(i=working2dnum[ii])>=0;ii++) {
    if((a_old[i]>0) && (a[i]==0)) { /* remove from model */
      pos=model->index[i]; 
      model->index[i]=-1;
      (model->sv_num)--;
      model->supvec[pos]=model->supvec[model->sv_num];
      model->alpha[pos]=model->alpha[model->sv_num];
      model->index[(model->supvec[pos])->docnum]=pos;
    }
    else if((a_old[i]==0) && (a[i]>0)) { /* add to model */
      model->supvec[model->sv_num]=docs[i];
      model->alpha[model->sv_num]=a[i]*(double)label[i];
      model->index[i]=model->sv_num;
      (model->sv_num)++;
    }
    else if(a_old[i]==a[i]) { /* nothing to do */
    }
    else {  /* just update alpha */
      model->alpha[model->index[i]]=a[i]*(double)label[i];
    }
      
    ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
    if((a_old[i]>=ex_c) && (a[i]<ex_c)) { 
      (model->at_upper_bound)--;
    }
    else if((a_old[i]<ex_c) && (a[i]>=ex_c)) { 
      (model->at_upper_bound)++;
    }

    if((!b_calculated) 
       && (a[i]>learn_parm->epsilon_a) && (a[i]<ex_c)) {   /* calculate b */
     	model->b=((double)label[i]*learn_parm->eps-c[i]+lin[i]); 
	/* model->b=(-(double)label[i]+lin[i]); */
	b_calculated=1;
    }
  }      

  /* No alpha in the working set not at bounds, so b was not
     calculated in the usual way. The following handles this special
     case. */
  if(learn_parm->biased_hyperplane 
     && (!b_calculated)
     && (model->sv_num-1 == model->at_upper_bound)) { 
    first_low=1;
    first_high=1;
    b_low=0;
    b_high=0;
    for(ii=0;(i=active2dnum[ii])>=0;ii++) {
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if(a[i]<ex_c) { 
	if(label[i]>0)  {
	  b_temp=-(learn_parm->eps-c[i]+lin[i]);
	  if((b_temp>b_low) || (first_low)) {
	    b_low=b_temp;
	    first_low=0;
	  }
	}
	else {
	  b_temp=-(-learn_parm->eps-c[i]+lin[i]);
	  if((b_temp<b_high) || (first_high)) {
	    b_high=b_temp;
	    first_high=0;
	  }
	}
      }
      else {
	if(label[i]<0)  {
	  b_temp=-(-learn_parm->eps-c[i]+lin[i]);
	  if((b_temp>b_low) || (first_low)) {
	    b_low=b_temp;
	    first_low=0;
	  }
	}
	else {
	  b_temp=-(learn_parm->eps-c[i]+lin[i]);
	  if((b_temp<b_high) || (first_high)) {
	    b_high=b_temp;
	    first_high=0;
	  }
	}
      }
    }
    if(first_high) {
      model->b=-b_low;
    }
    else if(first_low) {
      model->b=-b_high;
    }
    else {
      model->b=-(b_high+b_low)/2.0;  /* select b as the middle of range */
      /* printf("\nb_low=%f, b_high=%f,b=%f\n",b_low,b_high,model->b); */
    }
  }

  if(verbosity>=3) {
    printf("done\n"); fflush(stdout);
  }

  return(model->sv_num-1); /* have to substract one, since element 0 is empty*/
}

long check_optimality(MODEL *model, long int *label, long int *unlabeled, 
		      double *a, double *lin, double *c, long int totdoc, 
		      LEARN_PARM *learn_parm, double *maxdiff, 
		      double epsilon_crit_org, long int *misclassified, 
		      long int *inconsistent, long int *active2dnum,
		      long int *last_suboptimal_at, 
		      long int iteration, KERNEL_PARM *kernel_parm)
     /* Check KT-conditions */
{
  long i,ii,retrain;
  double dist,ex_c,target;

  if(kernel_parm->kernel_type == LINEAR) {  /* be optimistic */
    learn_parm->epsilon_shrink=-learn_parm->epsilon_crit+epsilon_crit_org;  
  }
  else {  /* be conservative */
    learn_parm->epsilon_shrink=learn_parm->epsilon_shrink*0.7+(*maxdiff)*0.3; 
  }
  retrain=0;
  (*maxdiff)=0;
  (*misclassified)=0;
  for(ii=0;(i=active2dnum[ii])>=0;ii++) {
    if((!inconsistent[i]) && label[i]) {
      dist=(lin[i]-model->b)*(double)label[i];/* 'distance' from
						 hyperplane*/
      target=-(learn_parm->eps-(double)label[i]*c[i]);
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if(dist <= 0) {       
	(*misclassified)++;  /* does not work due to deactivation of var */
      }
      if((a[i]>learn_parm->epsilon_a) && (dist > target)) {
	if((dist-target)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-target;
      }
      else if((a[i]<ex_c) && (dist < target)) {
	if((target-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=target-dist;
      }
      /* Count how long a variable was at lower/upper bound (and optimal).*/
      /* Variables, which were at the bound and optimal for a long */
      /* time are unlikely to become support vectors. In case our */
      /* cache is filled up, those variables are excluded to save */
      /* kernel evaluations. (See chapter 'Shrinking').*/ 
      if((a[i]>(learn_parm->epsilon_a)) 
	 && (a[i]<ex_c)) { 
	last_suboptimal_at[i]=iteration;         /* not at bound */
      }
      else if((a[i]<=(learn_parm->epsilon_a)) 
	      && (dist < (target+learn_parm->epsilon_shrink))) {
	last_suboptimal_at[i]=iteration;         /* not likely optimal */
      }
      else if((a[i]>=ex_c)
	      && (dist > (target-learn_parm->epsilon_shrink)))  { 
	last_suboptimal_at[i]=iteration;         /* not likely optimal */
      }
    }   
  }
  /* termination criterion */
  if((!retrain) && ((*maxdiff) > learn_parm->epsilon_crit)) {  
    retrain=1;
  }
  return(retrain);
}

long check_optimality_sharedslack(DOC **docs, MODEL *model, long int *label,
		      double *a, double *lin, double *c, double *slack,
		      double *alphaslack,
		      long int totdoc, 
		      LEARN_PARM *learn_parm, double *maxdiff, 
		      double epsilon_crit_org, long int *misclassified, 
		      long int *active2dnum,
		      long int *last_suboptimal_at, 
		      long int iteration, KERNEL_PARM *kernel_parm)
     /* Check KT-conditions */
{
  long i,ii,retrain;
  double dist,ex_c=0,target;

  if(kernel_parm->kernel_type == LINEAR) {  /* be optimistic */
    learn_parm->epsilon_shrink=-learn_parm->epsilon_crit+epsilon_crit_org;  
  }
  else {  /* be conservative */
    learn_parm->epsilon_shrink=learn_parm->epsilon_shrink*0.7+(*maxdiff)*0.3; 
  }

  retrain=0;
  (*maxdiff)=0;
  (*misclassified)=0;
  for(ii=0;(i=active2dnum[ii])>=0;ii++) {
    /* 'distance' from hyperplane*/
    dist=(lin[i]-model->b)*(double)label[i]+slack[docs[i]->slackid];
    target=-(learn_parm->eps-(double)label[i]*c[i]);
    ex_c=learn_parm->svm_c-learn_parm->epsilon_a;
    if((a[i]>learn_parm->epsilon_a) && (dist > target)) {
      if((dist-target)>(*maxdiff)) {  /* largest violation */
	(*maxdiff)=dist-target;
	if(verbosity>=5) printf("sid %ld: dist=%.2f, target=%.2f, slack=%.2f, a=%f, alphaslack=%f\n",docs[i]->slackid,dist,target,slack[docs[i]->slackid],a[i],alphaslack[docs[i]->slackid]);
	if(verbosity>=5) printf(" (single %f)\n",(*maxdiff));
      }
    }
    if((alphaslack[docs[i]->slackid]<ex_c) && (slack[docs[i]->slackid]>0)) {
      if((slack[docs[i]->slackid])>(*maxdiff)) { /* largest violation */
	(*maxdiff)=slack[docs[i]->slackid];
	if(verbosity>=5) printf("sid %ld: dist=%.2f, target=%.2f, slack=%.2f, a=%f, alphaslack=%f\n",docs[i]->slackid,dist,target,slack[docs[i]->slackid],a[i],alphaslack[docs[i]->slackid]);
	if(verbosity>=5) printf(" (joint %f)\n",(*maxdiff));
      }
    }
    /* Count how long a variable was at lower/upper bound (and optimal).*/
    /* Variables, which were at the bound and optimal for a long */
    /* time are unlikely to become support vectors. In case our */
    /* cache is filled up, those variables are excluded to save */
    /* kernel evaluations. (See chapter 'Shrinking').*/ 
    if((a[i]>(learn_parm->epsilon_a)) 
       && (a[i]<ex_c)) { 
      last_suboptimal_at[docs[i]->slackid]=iteration;  /* not at bound */
    }
    else if((a[i]<=(learn_parm->epsilon_a)) 
	    && (dist < (target+learn_parm->epsilon_shrink))) {
      last_suboptimal_at[docs[i]->slackid]=iteration;  /* not likely optimal */
    }
    else if((a[i]>=ex_c)
	    && (slack[docs[i]->slackid] < learn_parm->epsilon_shrink))  { 
      last_suboptimal_at[docs[i]->slackid]=iteration;  /* not likely optimal */
    }
  }   
  /* termination criterion */
  if((!retrain) && ((*maxdiff) > learn_parm->epsilon_crit)) {  
    retrain=1;
  }
  return(retrain);
}

void compute_shared_slacks(DOC **docs, long int *label, 
			   double *a, double *lin, 
			   double *c, long int *active2dnum,
			   LEARN_PARM *learn_parm, 
			   double *slack, double *alphaslack)
     /* compute the value of shared slacks and the joint alphas */
{
  long jj,i;
  double dist,target;

  for(jj=0;(i=active2dnum[jj])>=0;jj++) { /* clear slack variables */
    slack[docs[i]->slackid]=0.0;
    alphaslack[docs[i]->slackid]=0.0;
  }
  for(jj=0;(i=active2dnum[jj])>=0;jj++) { /* recompute slack variables */
    dist=(lin[i])*(double)label[i];
    target=-(learn_parm->eps-(double)label[i]*c[i]);
    if((target-dist) > slack[docs[i]->slackid])
      slack[docs[i]->slackid]=target-dist;
    alphaslack[docs[i]->slackid]+=a[i];
  }
}


long identify_inconsistent(double *a, long int *label, 
			   long int *unlabeled, long int totdoc, 
			   LEARN_PARM *learn_parm, 
			   long int *inconsistentnum, long int *inconsistent)
{
  long i,retrain;

  /* Throw out examples with multipliers at upper bound. This */
  /* corresponds to the -i 1 option. */
  /* ATTENTION: this is just a heuristic for finding a close */
  /*            to minimum number of examples to exclude to */
  /*            make the problem separable with desired margin */
  retrain=0;
  for(i=0;i<totdoc;i++) {
    if((!inconsistent[i]) && (!unlabeled[i]) 
       && (a[i]>=(learn_parm->svm_cost[i]-learn_parm->epsilon_a))) { 
	(*inconsistentnum)++;
	inconsistent[i]=1;  /* never choose again */
	retrain=2;          /* start over */
	if(verbosity>=3) {
	  printf("inconsistent(%ld)..",i); fflush(stdout);
	}
    }
  }
  return(retrain);
}

long identify_misclassified(double *lin, long int *label, 
			    long int *unlabeled, long int totdoc, 
			    MODEL *model, long int *inconsistentnum, 
			    long int *inconsistent)
{
  long i,retrain;
  double dist;

  /* Throw out misclassified examples. This */
  /* corresponds to the -i 2 option. */
  /* ATTENTION: this is just a heuristic for finding a close */
  /*            to minimum number of examples to exclude to */
  /*            make the problem separable with desired margin */
  retrain=0;
  for(i=0;i<totdoc;i++) {
    dist=(lin[i]-model->b)*(double)label[i]; /* 'distance' from hyperplane*/  
    if((!inconsistent[i]) && (!unlabeled[i]) && (dist <= 0)) { 
	(*inconsistentnum)++;
	inconsistent[i]=1;  /* never choose again */
	retrain=2;          /* start over */
	if(verbosity>=3) {
	  printf("inconsistent(%ld)..",i); fflush(stdout);
	}
    }
  }
  return(retrain);
}

long identify_one_misclassified(double *lin, long int *label, 
				long int *unlabeled, 
				long int totdoc, MODEL *model, 
				long int *inconsistentnum, 
				long int *inconsistent)
{
  long i,retrain,maxex=-1;
  double dist,maxdist=0;

  /* Throw out the 'most misclassified' example. This */
  /* corresponds to the -i 3 option. */
  /* ATTENTION: this is just a heuristic for finding a close */
  /*            to minimum number of examples to exclude to */
  /*            make the problem separable with desired margin */
  retrain=0;
  for(i=0;i<totdoc;i++) {
    if((!inconsistent[i]) && (!unlabeled[i])) {
      dist=(lin[i]-model->b)*(double)label[i];/* 'distance' from hyperplane*/  
      if(dist<maxdist) {
	maxdist=dist;
	maxex=i;
      }
    }
  }
  if(maxex>=0) {
    (*inconsistentnum)++;
    inconsistent[maxex]=1;  /* never choose again */
    retrain=2;          /* start over */
    if(verbosity>=3) {
      printf("inconsistent(%ld)..",i); fflush(stdout);
    }
  }
  return(retrain);
}

void update_linear_component(DOC **docs, long int *label, 
			     long int *active2dnum, double *a, 
			     double *a_old, long int *working2dnum, 
			     long int totdoc, long int totwords, 
			     KERNEL_PARM *kernel_parm, 
			     KERNEL_CACHE *kernel_cache, 
			     double *lin, CFLOAT *aicache, double *weights)
     /* keep track of the linear component */
     /* lin of the gradient etc. by updating */
     /* based on the change of the variables */
     /* in the current working set */
{
  register long i,ii,j,jj;
  register double tec;
  SVECTOR *f;

  if(kernel_parm->kernel_type==0) { /* special linear case */
    clear_vector_n(weights,totwords);
    for(ii=0;(i=working2dnum[ii])>=0;ii++) {
      if(a[i] != a_old[i]) {
	for(f=docs[i]->fvec;f;f=f->next)  
	  add_vector_ns(weights,f,
			f->factor*((a[i]-a_old[i])*(double)label[i]));
      }
    }
    for(jj=0;(j=active2dnum[jj])>=0;jj++) {
      for(f=docs[j]->fvec;f;f=f->next)  
	lin[j]+=f->factor*sprod_ns(weights,f);
    }
  }
  else {                            /* general case */
    for(jj=0;(i=working2dnum[jj])>=0;jj++) {
      if(a[i] != a_old[i]) {
	get_kernel_row(kernel_cache,docs,i,totdoc,active2dnum,aicache,
		       kernel_parm);
	for(ii=0;(j=active2dnum[ii])>=0;ii++) {
	  tec=aicache[j];
	  lin[j]+=(((a[i]*tec)-(a_old[i]*tec))*(double)label[i]);
	}
      }
    }
  }
}


long incorporate_unlabeled_examples(MODEL *model, long int *label, 
				    long int *inconsistent, 
				    long int *unlabeled, 
				    double *a, double *lin, 
				    long int totdoc, double *selcrit, 
				    long int *select, long int *key, 
				    long int transductcycle, 
				    KERNEL_PARM *kernel_parm, 
				    LEARN_PARM *learn_parm)
{
  long i,j,k,j1,j2,j3,j4,unsupaddnum1=0,unsupaddnum2=0;
  long pos,neg,upos,uneg,orgpos,orgneg,nolabel,newpos,newneg,allunlab;
  double dist,model_length,posratio,negratio;
  long check_every=2;
  double loss;
  static double switchsens=0.0,switchsensorg=0.0;
  double umin,umax,sumalpha;
  long imin=0,imax=0;
  static long switchnum=0;

  switchsens/=1.2;

  /* assumes that lin[] is up to date -> no inactive vars */

  orgpos=0;
  orgneg=0;
  newpos=0;
  newneg=0;
  nolabel=0;
  allunlab=0;
  for(i=0;i<totdoc;i++) {
    if(!unlabeled[i]) {
      if(label[i] > 0) {
	orgpos++;
      }
      else {
	orgneg++;
      }
    }
    else {
      allunlab++;
      if(unlabeled[i]) {
	if(label[i] > 0) {
	  newpos++;
	}
	else if(label[i] < 0) {
	  newneg++;
	}
      }
    }
    if(label[i]==0) {
      nolabel++;
    }
  }

  if(learn_parm->transduction_posratio >= 0) {
    posratio=learn_parm->transduction_posratio;
  }
  else {
    posratio=(double)orgpos/(double)(orgpos+orgneg); /* use ratio of pos/neg */
  }                                                  /* in training data */
  negratio=1.0-posratio;

  learn_parm->svm_costratio=1.0;                     /* global */
  if(posratio>0) {
    learn_parm->svm_costratio_unlab=negratio/posratio;
  }
  else {
    learn_parm->svm_costratio_unlab=1.0;
  }
  
  pos=0;
  neg=0;
  upos=0;
  uneg=0;
  for(i=0;i<totdoc;i++) {
    dist=(lin[i]-model->b);  /* 'distance' from hyperplane*/
    if(dist>0) {
      pos++;
    }
    else {
      neg++;
    }
    if(unlabeled[i]) {
      if(dist>0) {
	upos++;
      }
      else {
	uneg++;
      }
    }
    if((!unlabeled[i]) && (a[i]>(learn_parm->svm_cost[i]-learn_parm->epsilon_a))) {
      /*      printf("Ubounded %ld (class %ld, unlabeled %ld)\n",i,label[i],unlabeled[i]); */
    }
  }
  if(verbosity>=2) {
    printf("POS=%ld, ORGPOS=%ld, ORGNEG=%ld\n",pos,orgpos,orgneg);
    printf("POS=%ld, NEWPOS=%ld, NEWNEG=%ld\n",pos,newpos,newneg);
    printf("pos ratio = %f (%f).\n",(double)(upos)/(double)(allunlab),posratio);
    fflush(stdout);
  }

  if(transductcycle == 0) {
    j1=0; 
    j2=0;
    j4=0;
    for(i=0;i<totdoc;i++) {
      dist=(lin[i]-model->b);  /* 'distance' from hyperplane*/
      if((label[i]==0) && (unlabeled[i])) {
	selcrit[j4]=dist;
	key[j4]=i;
	j4++;
      }
    }
    unsupaddnum1=0;	
    unsupaddnum2=0;	
    select_top_n(selcrit,j4,select,(long)(allunlab*posratio+0.5));
    for(k=0;(k<(long)(allunlab*posratio+0.5));k++) {
      i=key[select[k]];
      label[i]=1;
      unsupaddnum1++;	
      j1++;
    }
    for(i=0;i<totdoc;i++) {
      if((label[i]==0) && (unlabeled[i])) {
	label[i]=-1;
	j2++;
	unsupaddnum2++;
      }
    }
    for(i=0;i<totdoc;i++) {  /* set upper bounds on vars */
      if(unlabeled[i]) {
	if(label[i] == 1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_costratio_unlab*learn_parm->svm_unlabbound;
	}
	else if(label[i] == -1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_unlabbound;
	}
      }
    }
    if(verbosity>=1) {
      /* printf("costratio %f, costratio_unlab %f, unlabbound %f\n",
	 learn_parm->svm_costratio,learn_parm->svm_costratio_unlab,
	 learn_parm->svm_unlabbound); */
      printf("Classifying unlabeled data as %ld POS / %ld NEG.\n",
	     unsupaddnum1,unsupaddnum2); 
      fflush(stdout);
    }
    if(verbosity >= 1) 
      printf("Retraining.");
    if(verbosity >= 2) printf("\n");
    return((long)3);
  }
  if((transductcycle % check_every) == 0) {
    if(verbosity >= 1) 
      printf("Retraining.");
    if(verbosity >= 2) printf("\n");
    j1=0;
    j2=0;
    unsupaddnum1=0;
    unsupaddnum2=0;
    for(i=0;i<totdoc;i++) {
      if((unlabeled[i] == 2)) {
	unlabeled[i]=1;
	label[i]=1;
	j1++;
	unsupaddnum1++;
      }
      else if((unlabeled[i] == 3)) {
	unlabeled[i]=1;
	label[i]=-1;
	j2++;
	unsupaddnum2++;
      }
    }
    for(i=0;i<totdoc;i++) {  /* set upper bounds on vars */
      if(unlabeled[i]) {
	if(label[i] == 1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_costratio_unlab*learn_parm->svm_unlabbound;
	}
	else if(label[i] == -1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_unlabbound;
	}
      }
    }

    if(verbosity>=2) {
      /* printf("costratio %f, costratio_unlab %f, unlabbound %f\n",
	     learn_parm->svm_costratio,learn_parm->svm_costratio_unlab,
	     learn_parm->svm_unlabbound); */
      printf("%ld positive -> Added %ld POS / %ld NEG unlabeled examples.\n",
	     upos,unsupaddnum1,unsupaddnum2); 
      fflush(stdout);
    }

    if(learn_parm->svm_unlabbound == 1) {
      learn_parm->epsilon_crit=0.001; /* do the last run right */
    }
    else {
      learn_parm->epsilon_crit=0.01; /* otherwise, no need to be so picky */
    }

    return((long)3);
  }
  else if(((transductcycle % check_every) < check_every)) { 
    model_length=0;
    sumalpha=0;
    loss=0;
    for(i=0;i<totdoc;i++) {
      model_length+=a[i]*label[i]*lin[i];
      sumalpha+=a[i];
      dist=(lin[i]-model->b);  /* 'distance' from hyperplane*/
      if((label[i]*dist)<(1.0-learn_parm->epsilon_crit)) {
	loss+=(1.0-(label[i]*dist))*learn_parm->svm_cost[i]; 
      }
    }
    model_length=sqrt(model_length); 
    if(verbosity>=2) {
      printf("Model-length = %f (%f), loss = %f, objective = %f\n",
	     model_length,sumalpha,loss,loss+0.5*model_length*model_length);
      fflush(stdout);
    }
    j1=0;
    j2=0;
    j3=0;
    j4=0;
    unsupaddnum1=0;	
    unsupaddnum2=0;	
    umin=99999;
    umax=-99999;
    j4=1;
    while(j4) {
      umin=99999;
      umax=-99999;
      for(i=0;(i<totdoc);i++) { 
	dist=(lin[i]-model->b);  
	if((label[i]>0) && (unlabeled[i]) && (!inconsistent[i]) 
	   && (dist<umin)) {
	  umin=dist;
	  imin=i;
	}
	if((label[i]<0) && (unlabeled[i])  && (!inconsistent[i]) 
	   && (dist>umax)) {
	  umax=dist;
	  imax=i;
	}
      }
      if((umin < (umax+switchsens-1E-4))) {
	j1++;
	j2++;
	unsupaddnum1++;	
	unlabeled[imin]=3;
	inconsistent[imin]=1;
	unsupaddnum2++;	
	unlabeled[imax]=2;
	inconsistent[imax]=1;
      }
      else
	j4=0;
      j4=0;
    }
    for(j=0;(j<totdoc);j++) {
      if(unlabeled[j] && (!inconsistent[j])) {
	if(label[j]>0) {
	  unlabeled[j]=2;
	}
	else if(label[j]<0) {
	  unlabeled[j]=3;
	}
	/* inconsistent[j]=1; */
	j3++;
      }
    }
    switchnum+=unsupaddnum1+unsupaddnum2;

    /* stop and print out current margin
       printf("switchnum %ld %ld\n",switchnum,kernel_parm->poly_degree);
       if(switchnum == 2*kernel_parm->poly_degree) {
       learn_parm->svm_unlabbound=1;
       }
       */

    if((!unsupaddnum1) && (!unsupaddnum2)) {
      if((learn_parm->svm_unlabbound>=1) && ((newpos+newneg) == allunlab)) {
	for(j=0;(j<totdoc);j++) {
	  inconsistent[j]=0;
	  if(unlabeled[j]) unlabeled[j]=1;
	}
	write_prediction(learn_parm->predfile,model,lin,a,unlabeled,label,
			 totdoc,learn_parm);  
	if(verbosity>=1)
	  printf("Number of switches: %ld\n",switchnum);
	return((long)0);
      }
      switchsens=switchsensorg;
      learn_parm->svm_unlabbound*=1.5;
      if(learn_parm->svm_unlabbound>1) {
	learn_parm->svm_unlabbound=1;
      }
      model->at_upper_bound=0; /* since upper bound increased */
      if(verbosity>=1) 
	printf("Increasing influence of unlabeled examples to %f%% .",
	       learn_parm->svm_unlabbound*100.0);
    }
    else if(verbosity>=1) {
      printf("%ld positive -> Switching labels of %ld POS / %ld NEG unlabeled examples.",
	     upos,unsupaddnum1,unsupaddnum2); 
      fflush(stdout);
    }

    if(verbosity >= 2) printf("\n");
    
    learn_parm->epsilon_crit=0.5; /* don't need to be so picky */

    for(i=0;i<totdoc;i++) {  /* set upper bounds on vars */
      if(unlabeled[i]) {
	if(label[i] == 1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_costratio_unlab*learn_parm->svm_unlabbound;
	}
	else if(label[i] == -1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_unlabbound;
	}
      }
    }

    return((long)2);
  }

  return((long)0); 
}

/*************************** Working set selection ***************************/

long select_next_qp_subproblem_grad(long int *label, 
				    long int *unlabeled, 
				    double *a, double *lin, 
				    double *c, long int totdoc, 
				    long int qp_size, 
				    LEARN_PARM *learn_parm, 
				    long int *inconsistent, 
				    long int *active2dnum, 
				    long int *working2dnum, 
				    double *selcrit, 
				    long int *select, 
				    KERNEL_CACHE *kernel_cache, 
				    long int cache_only,
				    long int *key, long int *chosen)
     /* Use the feasible direction approach to select the next
      qp-subproblem (see chapter 'Selecting a good working set'). If
      'cache_only' is true, then the variables are selected only among
      those for which the kernel evaluations are cached. */
{
  long choosenum,i,j,k,activedoc,inum,valid;
  double s;

  for(inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
  choosenum=0;
  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=-label[j];
    if(kernel_cache && cache_only) 
      valid=(kernel_cache->index[j]>=0);
    else
      valid=1;
    if(valid
       && (!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0)))
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j]))
      {
      selcrit[activedoc]=(double)label[j]*(learn_parm->eps-(double)label[j]*c[j]+(double)label[j]*lin[j]);
      /*      selcrit[activedoc]=(double)label[j]*(-1.0+(double)label[j]*lin[j]); */
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
    /* if(learn_parm->biased_hyperplane || (selcrit[select[k]] > 0)) { */
      i=key[select[k]];
      chosen[i]=1;
      working2dnum[inum+choosenum]=i;
      choosenum+=1;
      if(kernel_cache)
	kernel_cache_touch(kernel_cache,i); /* make sure it does not get
					       kicked out of cache */
      /* } */
  }

  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=label[j];
    if(kernel_cache && cache_only) 
      valid=(kernel_cache->index[j]>=0);
    else
      valid=1;
    if(valid
       && (!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0))) 
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j])) 
      {
      selcrit[activedoc]=-(double)label[j]*(learn_parm->eps-(double)label[j]*c[j]+(double)label[j]*lin[j]);
      /*  selcrit[activedoc]=-(double)(label[j]*(-1.0+(double)label[j]*lin[j])); */
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
    /* if(learn_parm->biased_hyperplane || (selcrit[select[k]] > 0)) { */
      i=key[select[k]];
      chosen[i]=1;
      working2dnum[inum+choosenum]=i;
      choosenum+=1;
      if(kernel_cache)
	kernel_cache_touch(kernel_cache,i); /* make sure it does not get
					       kicked out of cache */
      /* } */
  } 
  working2dnum[inum+choosenum]=-1; /* complete index */
  return(choosenum);
}

long select_next_qp_subproblem_rand(long int *label, 
				    long int *unlabeled, 
				    double *a, double *lin, 
				    double *c, long int totdoc, 
				    long int qp_size, 
				    LEARN_PARM *learn_parm, 
				    long int *inconsistent, 
				    long int *active2dnum, 
				    long int *working2dnum, 
				    double *selcrit, 
				    long int *select, 
				    KERNEL_CACHE *kernel_cache, 
				    long int *key, 
				    long int *chosen, 
				    long int iteration)
/* Use the feasible direction approach to select the next
   qp-subproblem (see section 'Selecting a good working set'). Chooses
   a feasible direction at (pseudo) random to help jump over numerical
   problem. */
{
  long choosenum,i,j,k,activedoc,inum;
  double s;

  for(inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
  choosenum=0;
  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=-label[j];
    if((!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0)))
       && (!inconsistent[j]) 
       && (label[j])
       && (!chosen[j])) {
      selcrit[activedoc]=(j+iteration) % totdoc;
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(kernel_cache,i); /* make sure it does not get kicked */
                                        /* out of cache */
  }

  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=label[j];
    if((!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0))) 
       && (!inconsistent[j]) 
       && (label[j])
       && (!chosen[j])) {
      selcrit[activedoc]=(j+iteration) % totdoc;
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(kernel_cache,i); /* make sure it does not get kicked */
                                        /* out of cache */
  } 
  working2dnum[inum+choosenum]=-1; /* complete index */
  return(choosenum);
}

long select_next_qp_slackset(DOC **docs, long int *label, 
			     double *a, double *lin, 
			     double *slack, double *alphaslack, 
			     double *c,
			     LEARN_PARM *learn_parm, 
			     long int *active2dnum, double *maxviol)
     /* returns the slackset with the largest internal violation */
{
  long i,ii,maxdiffid;
  double dist,target,maxdiff,ex_c;

  maxdiff=0;
  maxdiffid=0;
  for(ii=0;(i=active2dnum[ii])>=0;ii++) {
    ex_c=learn_parm->svm_c-learn_parm->epsilon_a;
    if(alphaslack[docs[i]->slackid] >= ex_c) {
      dist=(lin[i])*(double)label[i]+slack[docs[i]->slackid]; /* distance */
      target=-(learn_parm->eps-(double)label[i]*c[i]); /* rhs of constraint */
      if((a[i]>learn_parm->epsilon_a) && (dist > target)) {
	if((dist-target)>maxdiff) { /* largest violation */
	  maxdiff=dist-target;
	  maxdiffid=docs[i]->slackid;
	}
      }
    }
  }
  (*maxviol)=maxdiff;
  return(maxdiffid);
}


void select_top_n(double *selcrit, long int range, long int *select, 
		  long int n)
{
  register long i,j;

  for(i=0;(i<n) && (i<range);i++) { /* Initialize with the first n elements */
    for(j=i;j>=0;j--) {
      if((j>0) && (selcrit[select[j-1]]<selcrit[i])){
	select[j]=select[j-1];
      }
      else {
	select[j]=i;
	j=-1;
      }
    }
  }
  if(n>0) {
    for(i=n;i<range;i++) {  
      if(selcrit[i]>selcrit[select[n-1]]) {
	for(j=n-1;j>=0;j--) {
	  if((j>0) && (selcrit[select[j-1]]<selcrit[i])) {
	    select[j]=select[j-1];
	  }
	  else {
	    select[j]=i;
	    j=-1;
	  }
	}
      }
    }
  }
}      
      

/******************************** Shrinking  *********************************/

void init_shrink_state(SHRINK_STATE *shrink_state, long int totdoc, 
		       long int maxhistory)
{
  long i;

  shrink_state->deactnum=0;
  shrink_state->active = (long *)my_malloc(sizeof(long)*totdoc);
  shrink_state->inactive_since = (long *)my_malloc(sizeof(long)*totdoc);
  shrink_state->a_history = (double **)my_malloc(sizeof(double *)*maxhistory);
  shrink_state->maxhistory=maxhistory;
  shrink_state->last_lin = (double *)my_malloc(sizeof(double)*totdoc);
  shrink_state->last_a = (double *)my_malloc(sizeof(double)*totdoc);

  for(i=0;i<totdoc;i++) { 
    shrink_state->active[i]=1;
    shrink_state->inactive_since[i]=0;
    shrink_state->last_a[i]=0;
    shrink_state->last_lin[i]=0;
  }
}

void shrink_state_cleanup(SHRINK_STATE *shrink_state)
{
  free(shrink_state->active);
  free(shrink_state->inactive_since);
  if(shrink_state->deactnum > 0) 
    free(shrink_state->a_history[shrink_state->deactnum-1]);
  free(shrink_state->a_history);
  free(shrink_state->last_a);
  free(shrink_state->last_lin);
}

long shrink_problem(DOC **docs,
		    LEARN_PARM *learn_parm, 
		    SHRINK_STATE *shrink_state, 
		    KERNEL_PARM *kernel_parm,
		    long int *active2dnum, 
		    long int *last_suboptimal_at, 
		    long int iteration, 
		    long int totdoc, 
		    long int minshrink, 
		    double *a, 
		    long int *inconsistent)
     /* Shrink some variables away.  Do the shrinking only if at least
        minshrink variables can be removed. */
{
  long i,ii,change,activenum,lastiter;
  double *a_old;
  
  activenum=0;
  change=0;
  for(ii=0;active2dnum[ii]>=0;ii++) {
    i=active2dnum[ii];
    activenum++;
    if(learn_parm->sharedslack)
      lastiter=last_suboptimal_at[docs[i]->slackid];
    else
      lastiter=last_suboptimal_at[i];
    if(((iteration-lastiter) > learn_parm->svm_iter_to_shrink) 
       || (inconsistent[i])) {
      change++;
    }
  }
  if((change>=minshrink) /* shrink only if sufficiently many candidates */
     && (shrink_state->deactnum<shrink_state->maxhistory)) { /* and enough memory */
    /* Shrink problem by removing those variables which are */
    /* optimal at a bound for a minimum number of iterations */
    if(verbosity>=2) {
      printf(" Shrinking..."); fflush(stdout);
    }
    if(kernel_parm->kernel_type != LINEAR) { /*  non-linear case save alphas */
      a_old=(double *)my_malloc(sizeof(double)*totdoc);
      shrink_state->a_history[shrink_state->deactnum]=a_old;
      for(i=0;i<totdoc;i++) {
	a_old[i]=a[i];
      }
    }
    for(ii=0;active2dnum[ii]>=0;ii++) {
      i=active2dnum[ii];
      if(learn_parm->sharedslack)
	lastiter=last_suboptimal_at[docs[i]->slackid];
      else
	lastiter=last_suboptimal_at[i];
      if(((iteration-lastiter) > learn_parm->svm_iter_to_shrink) 
	 || (inconsistent[i])) {
	shrink_state->active[i]=0;
	shrink_state->inactive_since[i]=shrink_state->deactnum;
      }
    }
    activenum=compute_index(shrink_state->active,totdoc,active2dnum);
    shrink_state->deactnum++;
    if(kernel_parm->kernel_type == LINEAR) { 
      shrink_state->deactnum=0;
    }
    if(verbosity>=2) {
      printf("done.\n"); fflush(stdout);
      printf(" Number of inactive variables = %ld\n",totdoc-activenum);
    }
  }
  return(activenum);
} 


void reactivate_inactive_examples(long int *label, 
				  long int *unlabeled, 
				  double *a, 
				  SHRINK_STATE *shrink_state, 
				  double *lin, 
				  double *c, 
				  long int totdoc, 
				  long int totwords, 
				  long int iteration, 
				  LEARN_PARM *learn_parm, 
				  long int *inconsistent, 
				  DOC **docs, 
				  KERNEL_PARM *kernel_parm, 
				  KERNEL_CACHE *kernel_cache, 
				  MODEL *model, 
				  CFLOAT *aicache, 
				  double *weights, 
				  double *maxdiff)
     /* Make all variables active again which had been removed by
        shrinking. */
     /* Computes lin for those variables from scratch. */
{
  register long i,j,ii,jj,t,*changed2dnum,*inactive2dnum;
  long *changed,*inactive;
  register double kernel_val,*a_old,dist;
  double ex_c,target;
  SVECTOR *f;

  if(kernel_parm->kernel_type == LINEAR) { /* special linear case */
    a_old=shrink_state->last_a;    
    clear_vector_n(weights,totwords);
    for(i=0;i<totdoc;i++) {
      if(a[i] != a_old[i]) {
	for(f=docs[i]->fvec;f;f=f->next)  
	  add_vector_ns(weights,f,
			f->factor*((a[i]-a_old[i])*(double)label[i]));
	a_old[i]=a[i];
      }
    }
    for(i=0;i<totdoc;i++) {
      if(!shrink_state->active[i]) {
	for(f=docs[i]->fvec;f;f=f->next)  
	  lin[i]=shrink_state->last_lin[i]+f->factor*sprod_ns(weights,f);
      }
      shrink_state->last_lin[i]=lin[i];
    }
  }
  else {
    changed=(long *)my_malloc(sizeof(long)*totdoc);
    changed2dnum=(long *)my_malloc(sizeof(long)*(totdoc+11));
    inactive=(long *)my_malloc(sizeof(long)*totdoc);
    inactive2dnum=(long *)my_malloc(sizeof(long)*(totdoc+11));
    for(t=shrink_state->deactnum-1;(t>=0) && shrink_state->a_history[t];t--) {
      if(verbosity>=2) {
	printf("%ld..",t); fflush(stdout);
      }
      a_old=shrink_state->a_history[t];    
      for(i=0;i<totdoc;i++) {
	inactive[i]=((!shrink_state->active[i]) 
		     && (shrink_state->inactive_since[i] == t));
	changed[i]= (a[i] != a_old[i]);
      }
      compute_index(inactive,totdoc,inactive2dnum);
      compute_index(changed,totdoc,changed2dnum);
      
      for(ii=0;(i=changed2dnum[ii])>=0;ii++) {
	get_kernel_row(kernel_cache,docs,i,totdoc,inactive2dnum,aicache,
		       kernel_parm);
	for(jj=0;(j=inactive2dnum[jj])>=0;jj++) {
	  kernel_val=aicache[j];
	  lin[j]+=(((a[i]*kernel_val)-(a_old[i]*kernel_val))*(double)label[i]);
	}
      }
    }
    free(changed);
    free(changed2dnum);
    free(inactive);
    free(inactive2dnum);
  }
  (*maxdiff)=0;
  for(i=0;i<totdoc;i++) {
    shrink_state->inactive_since[i]=shrink_state->deactnum-1;
    if(!inconsistent[i]) {
      dist=(lin[i]-model->b)*(double)label[i];
      target=-(learn_parm->eps-(double)label[i]*c[i]);
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if((a[i]>learn_parm->epsilon_a) && (dist > target)) {
	if((dist-target)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-target;
      }
      else if((a[i]<ex_c) && (dist < target)) {
	if((target-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=target-dist;
      }
      if((a[i]>(0+learn_parm->epsilon_a)) 
	 && (a[i]<ex_c)) { 
	shrink_state->active[i]=1;                         /* not at bound */
      }
      else if((a[i]<=(0+learn_parm->epsilon_a)) && (dist < (target+learn_parm->epsilon_shrink))) {
	shrink_state->active[i]=1;
      }
      else if((a[i]>=ex_c)
	      && (dist > (target-learn_parm->epsilon_shrink))) {
	shrink_state->active[i]=1;
      }
      else if(learn_parm->sharedslack) { /* make all active when sharedslack */
	shrink_state->active[i]=1;
      }
    }
  }
  if(kernel_parm->kernel_type != LINEAR) { /* update history for non-linear */
    for(i=0;i<totdoc;i++) {
      (shrink_state->a_history[shrink_state->deactnum-1])[i]=a[i];
    }
    for(t=shrink_state->deactnum-2;(t>=0) && shrink_state->a_history[t];t--) {
      free(shrink_state->a_history[t]);
      shrink_state->a_history[t]=0;
    }
  }
}

/****************************** Cache handling *******************************/

void get_kernel_row(KERNEL_CACHE *kernel_cache, DOC **docs, 
		    long int docnum, long int totdoc, 
		    long int *active2dnum, CFLOAT *buffer, 
		    KERNEL_PARM *kernel_parm)
     /* Get's a row of the matrix of kernel values This matrix has the
      same form as the Hessian, just that the elements are not
      multiplied by */
     /* y_i * y_j * a_i * a_j */
     /* Takes the values from the cache if available. */
{
  register long i,j,start;
  DOC *ex;

  ex=docs[docnum];

  if(kernel_cache->index[docnum] != -1) { /* row is cached? */
    kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; /* lru */
    start=kernel_cache->activenum*kernel_cache->index[docnum];
    for(i=0;(j=active2dnum[i])>=0;i++) {
      if(kernel_cache->totdoc2active[j] >= 0) { /* column is cached? */
	buffer[j]=kernel_cache->buffer[start+kernel_cache->totdoc2active[j]];
      }
      else {
	buffer[j]=(CFLOAT)kernel(kernel_parm,ex,docs[j]);
      }
    }
  }
  else {
    for(i=0;(j=active2dnum[i])>=0;i++) {
      buffer[j]=(CFLOAT)kernel(kernel_parm,ex,docs[j]);
    }
  }
}


void cache_kernel_row(KERNEL_CACHE *kernel_cache, DOC **docs, 
		      long int m, KERNEL_PARM *kernel_parm)
     /* Fills cache for the row m */
{
  register DOC *ex;
  register long j,k,l;
  register CFLOAT *cache;

  if(!kernel_cache_check(kernel_cache,m)) {  /* not cached yet*/
    cache = kernel_cache_clean_and_malloc(kernel_cache,m);
    if(cache) {
      l=kernel_cache->totdoc2active[m];
      ex=docs[m];
      for(j=0;j<kernel_cache->activenum;j++) {  /* fill cache */
	k=kernel_cache->active2totdoc[j];
	if((kernel_cache->index[k] != -1) && (l != -1) && (k != m)) {
	  cache[j]=kernel_cache->buffer[kernel_cache->activenum
				       *kernel_cache->index[k]+l];
	}
	else {
	  cache[j]=kernel(kernel_parm,ex,docs[k]);
	} 
      }
    }
    else {
      perror("Error: Kernel cache full! => increase cache size");
    }
  }
}

 
void cache_multiple_kernel_rows(KERNEL_CACHE *kernel_cache, DOC **docs, 
				long int *key, long int varnum, 
				KERNEL_PARM *kernel_parm)
     /* Fills cache for the rows in key */
{
  register long i;

  for(i=0;i<varnum;i++) {  /* fill up kernel cache */
    cache_kernel_row(kernel_cache,docs,key[i],kernel_parm);
  }
}

 
void kernel_cache_shrink(KERNEL_CACHE *kernel_cache, long int totdoc, 
			 long int numshrink, long int *after)
     /* Remove numshrink columns in the cache which correspond to
        examples marked 0 in after. */
{
  register long i,j,jj,from=0,to=0,scount;  
  long *keep;

  if(verbosity>=2) {
    printf(" Reorganizing cache..."); fflush(stdout);
  }

  keep=(long *)my_malloc(sizeof(long)*totdoc);
  for(j=0;j<totdoc;j++) {
    keep[j]=1;
  }
  scount=0;
  for(jj=0;(jj<kernel_cache->activenum) && (scount<numshrink);jj++) {
    j=kernel_cache->active2totdoc[jj];
    if(!after[j]) {
      scount++;
      keep[j]=0;
    }
  }

  for(i=0;i<kernel_cache->max_elems;i++) {
    for(jj=0;jj<kernel_cache->activenum;jj++) {
      j=kernel_cache->active2totdoc[jj];
      if(!keep[j]) {
	from++;
      }
      else {
	kernel_cache->buffer[to]=kernel_cache->buffer[from];
	to++;
	from++;
      }
    }
  }

  kernel_cache->activenum=0;
  for(j=0;j<totdoc;j++) {
    if((keep[j]) && (kernel_cache->totdoc2active[j] != -1)) {
      kernel_cache->active2totdoc[kernel_cache->activenum]=j;
      kernel_cache->totdoc2active[j]=kernel_cache->activenum;
      kernel_cache->activenum++;
    }
    else {
      kernel_cache->totdoc2active[j]=-1;
    }
  }

  kernel_cache->max_elems=(long)(kernel_cache->buffsize/kernel_cache->activenum);
  if(kernel_cache->max_elems>totdoc) {
    kernel_cache->max_elems=totdoc;
  }

  free(keep);

  if(verbosity>=2) {
    printf("done.\n"); fflush(stdout);
    printf(" Cache-size in rows = %ld\n",kernel_cache->max_elems);
  }
}

KERNEL_CACHE *kernel_cache_init(long int totdoc, long int buffsize)
{
  long i;
  KERNEL_CACHE *kernel_cache;

  kernel_cache=(KERNEL_CACHE *)my_malloc(sizeof(KERNEL_CACHE));
  kernel_cache->index = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->occu = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->lru = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->invindex = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->active2totdoc = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->totdoc2active = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->buffer = (CFLOAT *)my_malloc((size_t)(buffsize)*1024*1024);

  kernel_cache->buffsize=(long)(buffsize/sizeof(CFLOAT)*1024*1024);

  kernel_cache->max_elems=(long)(kernel_cache->buffsize/totdoc);
  if(kernel_cache->max_elems>totdoc) {
    kernel_cache->max_elems=totdoc;
  }

  if(verbosity>=2) {
    printf(" Cache-size in rows = %ld\n",kernel_cache->max_elems);
    printf(" Kernel evals so far: %ld\n",kernel_cache_statistic);    
  }

  kernel_cache->elems=0;   /* initialize cache */
  for(i=0;i<totdoc;i++) {
    kernel_cache->index[i]=-1;
    kernel_cache->lru[i]=0;
  }
  for(i=0;i<totdoc;i++) {
    kernel_cache->occu[i]=0;
    kernel_cache->invindex[i]=-1;
  }

  kernel_cache->activenum=totdoc;;
  for(i=0;i<totdoc;i++) {
      kernel_cache->active2totdoc[i]=i;
      kernel_cache->totdoc2active[i]=i;
  }

  kernel_cache->time=0;  

  return(kernel_cache);
} 

void kernel_cache_reset_lru(KERNEL_CACHE *kernel_cache)
{
  long maxlru=0,k;
  
  for(k=0;k<kernel_cache->max_elems;k++) {
    if(maxlru < kernel_cache->lru[k]) 
      maxlru=kernel_cache->lru[k];
  }
  for(k=0;k<kernel_cache->max_elems;k++) {
      kernel_cache->lru[k]-=maxlru;
  }
}

void kernel_cache_cleanup(KERNEL_CACHE *kernel_cache)
{
  free(kernel_cache->index);
  free(kernel_cache->occu);
  free(kernel_cache->lru);
  free(kernel_cache->invindex);
  free(kernel_cache->active2totdoc);
  free(kernel_cache->totdoc2active);
  free(kernel_cache->buffer);
  free(kernel_cache);
}

long kernel_cache_malloc(KERNEL_CACHE *kernel_cache)
{
  long i;

  if(kernel_cache_space_available(kernel_cache)) {
    for(i=0;i<kernel_cache->max_elems;i++) {
      if(!kernel_cache->occu[i]) {
	kernel_cache->occu[i]=1;
	kernel_cache->elems++;
	return(i);
      }
    }
  }
  return(-1);
}

void kernel_cache_free(KERNEL_CACHE *kernel_cache, long int i)
{
  kernel_cache->occu[i]=0;
  kernel_cache->elems--;
}

long kernel_cache_free_lru(KERNEL_CACHE *kernel_cache) 
     /* remove least recently used cache element */
{                                     
  register long k,least_elem=-1,least_time;

  least_time=kernel_cache->time+1;
  for(k=0;k<kernel_cache->max_elems;k++) {
    if(kernel_cache->invindex[k] != -1) {
      if(kernel_cache->lru[k]<least_time) {
	least_time=kernel_cache->lru[k];
	least_elem=k;
      }
    }
  }
  if(least_elem != -1) {
    kernel_cache_free(kernel_cache,least_elem);
    kernel_cache->index[kernel_cache->invindex[least_elem]]=-1;
    kernel_cache->invindex[least_elem]=-1;
    return(1);
  }
  return(0);
}


CFLOAT *kernel_cache_clean_and_malloc(KERNEL_CACHE *kernel_cache, 
				      long int docnum)
     /* Get a free cache entry. In case cache is full, the lru element
        is removed. */
{
  long result;
  if((result = kernel_cache_malloc(kernel_cache)) == -1) {
    if(kernel_cache_free_lru(kernel_cache)) {
      result = kernel_cache_malloc(kernel_cache);
    }
  }
  kernel_cache->index[docnum]=result;
  if(result == -1) {
    return(0);
  }
  kernel_cache->invindex[result]=docnum;
  kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; /* lru */
  return((CFLOAT *)((long)kernel_cache->buffer
		    +(kernel_cache->activenum*sizeof(CFLOAT)*
		      kernel_cache->index[docnum])));
}

long kernel_cache_touch(KERNEL_CACHE *kernel_cache, long int docnum)
     /* Update lru time to avoid removal from cache. */
{
  if(kernel_cache && kernel_cache->index[docnum] != -1) {
    kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; /* lru */
    return(1);
  }
  return(0);
}
  
long kernel_cache_check(KERNEL_CACHE *kernel_cache, long int docnum)
     /* Is that row cached? */
{
  return(kernel_cache->index[docnum] != -1);
}
  
long kernel_cache_space_available(KERNEL_CACHE *kernel_cache)
     /* Is there room for one more row? */
{
  return(kernel_cache->elems < kernel_cache->max_elems);
}
  
/************************** Compute estimates ******************************/

void compute_xa_estimates(MODEL *model, long int *label, 
			  long int *unlabeled, long int totdoc, 
			  DOC **docs, double *lin, double *a, 
			  KERNEL_PARM *kernel_parm, 
			  LEARN_PARM *learn_parm, double *error, 
			  double *recall, double *precision) 
     /* Computes xa-estimate of error rate, recall, and precision. See
        T. Joachims, Estimating the Generalization Performance of an SVM
        Efficiently, IMCL, 2000. */
{
  long i,looerror,looposerror,loonegerror;
  long totex,totposex;
  double xi,r_delta,r_delta_sq,sim=0;
  long *sv2dnum=NULL,*sv=NULL,svnum;

  r_delta=estimate_r_delta(docs,totdoc,kernel_parm); 
  r_delta_sq=r_delta*r_delta;

  looerror=0;
  looposerror=0;
  loonegerror=0;
  totex=0;
  totposex=0;
  svnum=0;

  if(learn_parm->xa_depth > 0) {
    sv = (long *)my_malloc(sizeof(long)*(totdoc+11));
    for(i=0;i<totdoc;i++) 
      sv[i]=0;
    for(i=1;i<model->sv_num;i++) 
      if(a[model->supvec[i]->docnum] 
	 < (learn_parm->svm_cost[model->supvec[i]->docnum]
	    -learn_parm->epsilon_a)) {
	sv[model->supvec[i]->docnum]=1;
	svnum++;
      }
    sv2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
    clear_index(sv2dnum);
    compute_index(sv,totdoc,sv2dnum);
  }

  for(i=0;i<totdoc;i++) {
    if(unlabeled[i]) {
      /* ignore it */
    }
    else {
      xi=1.0-((lin[i]-model->b)*(double)label[i]);
      if(xi<0) xi=0;
      if(label[i]>0) {
	totposex++;
      }
      if((learn_parm->rho*a[i]*r_delta_sq+xi) >= 1.0) {
	if(learn_parm->xa_depth > 0) {  /* makes assumptions */
	  sim=distribute_alpha_t_greedily(sv2dnum,svnum,docs,a,i,label,
					  kernel_parm,learn_parm,
		            (double)((1.0-xi-a[i]*r_delta_sq)/(2.0*a[i])));
	}
	if((learn_parm->xa_depth == 0) || 
	   ((a[i]*kernel(kernel_parm,docs[i],docs[i])+a[i]*2.0*sim+xi) >= 1.0)) { 
	  looerror++;
	  if(label[i]>0) {
	    looposerror++;
	  }
	  else {
	    loonegerror++;
	  }
	}
      }
      totex++;
    }
  }

  (*error)=((double)looerror/(double)totex)*100.0;
  (*recall)=(1.0-(double)looposerror/(double)totposex)*100.0;
  (*precision)=(((double)totposex-(double)looposerror)
    /((double)totposex-(double)looposerror+(double)loonegerror))*100.0;

  free(sv);
  free(sv2dnum);
}


double distribute_alpha_t_greedily(long int *sv2dnum, long int svnum, 
				   DOC **docs, double *a, 
				   long int docnum, 
				   long int *label, 
				   KERNEL_PARM *kernel_parm, 
				   LEARN_PARM *learn_parm, double thresh)
     /* Experimental Code improving plain XiAlpha Estimates by
	computing a better bound using a greedy optimzation strategy. */
{
  long best_depth=0;
  long i,j,k,d,skip,allskip;
  double best,best_val[101],val,init_val_sq,init_val_lin;
  long best_ex[101];
  CFLOAT *cache,*trow;

  cache=(CFLOAT *)my_malloc(sizeof(CFLOAT)*learn_parm->xa_depth*svnum);
  trow = (CFLOAT *)my_malloc(sizeof(CFLOAT)*svnum);

  for(k=0;k<svnum;k++) {
    trow[k]=kernel(kernel_parm,docs[docnum],docs[sv2dnum[k]]);
  }

  init_val_sq=0;
  init_val_lin=0;
  best=0;

  for(d=0;d<learn_parm->xa_depth;d++) {
    allskip=1;
    if(d>=1) {
      init_val_sq+=cache[best_ex[d-1]+svnum*(d-1)]; 
      for(k=0;k<d-1;k++) {
        init_val_sq+=2.0*cache[best_ex[k]+svnum*(d-1)]; 
      }
      init_val_lin+=trow[best_ex[d-1]]; 
    }
    for(i=0;i<svnum;i++) {
      skip=0;
      if(sv2dnum[i] == docnum) skip=1;
      for(j=0;j<d;j++) {
	if(i == best_ex[j]) skip=1;
      }

      if(!skip) {
	val=init_val_sq;
	if(kernel_parm->kernel_type == LINEAR) 
	  val+=docs[sv2dnum[i]]->fvec->twonorm_sq;
	else
	  val+=kernel(kernel_parm,docs[sv2dnum[i]],docs[sv2dnum[i]]);
	for(j=0;j<d;j++) {
	  val+=2.0*cache[i+j*svnum];
	}
	val*=(1.0/(2.0*(d+1.0)*(d+1.0)));
	val-=((init_val_lin+trow[i])/(d+1.0));

	if(allskip || (val < best_val[d])) {
	  best_val[d]=val;
	  best_ex[d]=i;
	}
	allskip=0;
	if(val < thresh) {
	  i=svnum;
	  /*	  printf("EARLY"); */
	}
      }
    }
    if(!allskip) {
      for(k=0;k<svnum;k++) {
	  cache[d*svnum+k]=kernel(kernel_parm,
				  docs[sv2dnum[best_ex[d]]],
				  docs[sv2dnum[k]]);
      }
    }
    if((!allskip) && ((best_val[d] < best) || (d == 0))) {
      best=best_val[d];
      best_depth=d;
    }
    if(allskip || (best < thresh)) {
      d=learn_parm->xa_depth;
    }
  }    

  free(cache);
  free(trow);

  /*  printf("Distribute[%ld](%ld)=%f, ",docnum,best_depth,best); */
  return(best);
}


void estimate_transduction_quality(MODEL *model, long int *label, 
				   long int *unlabeled, 
				   long int totdoc, DOC **docs, double *lin) 
     /* Loo-bound based on observation that loo-errors must have an
	equal distribution in both training and test examples, given
	that the test examples are classified correctly. Compare
	chapter "Constraints on the Transductive Hyperplane" in my
	Dissertation. */
{
  long i,j,l=0,ulab=0,lab=0,labpos=0,labneg=0,ulabpos=0,ulabneg=0,totulab=0;
  double totlab=0,totlabpos=0,totlabneg=0,labsum=0,ulabsum=0;
  double r_delta,r_delta_sq,xi,xisum=0,asum=0;

  r_delta=estimate_r_delta(docs,totdoc,&(model->kernel_parm)); 
  r_delta_sq=r_delta*r_delta;

  for(j=0;j<totdoc;j++) {
    if(unlabeled[j]) {
      totulab++;
    }
    else {
      totlab++;
      if(label[j] > 0) 
	totlabpos++;
      else 
	totlabneg++;
    }
  }
  for(j=1;j<model->sv_num;j++) {
    i=model->supvec[j]->docnum;
    xi=1.0-((lin[i]-model->b)*(double)label[i]);
    if(xi<0) xi=0;

    xisum+=xi;
    asum+=fabs(model->alpha[j]);
    if(unlabeled[i]) {
      ulabsum+=(fabs(model->alpha[j])*r_delta_sq+xi);
    }
    else {
      labsum+=(fabs(model->alpha[j])*r_delta_sq+xi);
    }
    if((fabs(model->alpha[j])*r_delta_sq+xi) >= 1) { 
      l++;
      if(unlabeled[model->supvec[j]->docnum]) {
	ulab++;
	if(model->alpha[j] > 0) 
	  ulabpos++;
	else 
	  ulabneg++;
      }
      else {
	lab++;
	if(model->alpha[j] > 0) 
	  labpos++;
	else 
	  labneg++;
      }
    }
  }
  printf("xacrit>=1: labeledpos=%.5f labeledneg=%.5f default=%.5f\n",(double)labpos/(double)totlab*100.0,(double)labneg/(double)totlab*100.0,(double)totlabpos/(double)(totlab)*100.0);
  printf("xacrit>=1: unlabelpos=%.5f unlabelneg=%.5f\n",(double)ulabpos/(double)totulab*100.0,(double)ulabneg/(double)totulab*100.0);
  printf("xacrit>=1: labeled=%.5f unlabled=%.5f all=%.5f\n",(double)lab/(double)totlab*100.0,(double)ulab/(double)totulab*100.0,(double)l/(double)(totdoc)*100.0);
  printf("xacritsum: labeled=%.5f unlabled=%.5f all=%.5f\n",(double)labsum/(double)totlab*100.0,(double)ulabsum/(double)totulab*100.0,(double)(labsum+ulabsum)/(double)(totdoc)*100.0);
  printf("r_delta_sq=%.5f xisum=%.5f asum=%.5f\n",r_delta_sq,xisum,asum);
}

double estimate_margin_vcdim(MODEL *model, double w, double R, 
			     KERNEL_PARM *kernel_parm) 
     /* optional: length of model vector in feature space */
     /* optional: radius of ball containing the data */
{
  double h;

  /* follows chapter 5.6.4 in [Vapnik/95] */

  if(w<0) {
    w=model_length_s(model,kernel_parm);
  }
  if(R<0) {
    R=estimate_sphere(model,kernel_parm); 
  }
  h = w*w * R*R +1; 
  return(h);
}

double estimate_sphere(MODEL *model, KERNEL_PARM *kernel_parm) 
                          /* Approximates the radius of the ball containing */
                          /* the support vectors by bounding it with the */
{                         /* length of the longest support vector. This is */
  register long j;        /* pretty good for text categorization, since all */
  double xlen,maxxlen=0;  /* documents have feature vectors of length 1. It */
  DOC *nulldoc;           /* assumes that the center of the ball is at the */
  WORD nullword;          /* origin of the space. */

  nullword.wnum=0;
  nulldoc=create_example(-2,0,0,0.0,create_svector(&nullword,"",1.0)); 

  for(j=1;j<model->sv_num;j++) {
    xlen=sqrt(kernel(kernel_parm,model->supvec[j],model->supvec[j])
	      -2*kernel(kernel_parm,model->supvec[j],nulldoc)
	      +kernel(kernel_parm,nulldoc,nulldoc));
    if(xlen>maxxlen) {
      maxxlen=xlen;
    }
  }

  free_example(nulldoc,1);
  return(maxxlen);
}

double estimate_r_delta(DOC **docs, long int totdoc, KERNEL_PARM *kernel_parm)
{
  long i;
  double maxxlen,xlen;
  DOC *nulldoc;           /* assumes that the center of the ball is at the */
  WORD nullword;          /* origin of the space. */

  nullword.wnum=0;
  nulldoc=create_example(-2,0,0,0.0,create_svector(&nullword,"",1.0)); 

  maxxlen=0;
  for(i=0;i<totdoc;i++) {
    xlen=sqrt(kernel(kernel_parm,docs[i],docs[i])
	      -2*kernel(kernel_parm,docs[i],nulldoc)
	      +kernel(kernel_parm,nulldoc,nulldoc));
    if(xlen>maxxlen) {
      maxxlen=xlen;
    }
  }

  free_example(nulldoc,1);
  return(maxxlen);
}

double estimate_r_delta_average(DOC **docs, long int totdoc, 
				KERNEL_PARM *kernel_parm)
{
  long i;
  double avgxlen;
  DOC *nulldoc;           /* assumes that the center of the ball is at the */
  WORD nullword;          /* origin of the space. */

  nullword.wnum=0;
  nulldoc=create_example(-2,0,0,0.0,create_svector(&nullword,"",1.0)); 

  avgxlen=0;
  for(i=0;i<totdoc;i++) {
    avgxlen+=sqrt(kernel(kernel_parm,docs[i],docs[i])
		  -2*kernel(kernel_parm,docs[i],nulldoc)
		  +kernel(kernel_parm,nulldoc,nulldoc));
  }

  free_example(nulldoc,1);
  return(avgxlen/totdoc);
}

double length_of_longest_document_vector(DOC **docs, long int totdoc, 
					 KERNEL_PARM *kernel_parm)
{
  long i;
  double maxxlen,xlen;

  maxxlen=0;
  for(i=0;i<totdoc;i++) {
    xlen=sqrt(kernel(kernel_parm,docs[i],docs[i]));
    if(xlen>maxxlen) {
      maxxlen=xlen;
    }
  }

  return(maxxlen);
}

/****************************** IO-handling **********************************/

void write_prediction(char *predfile, MODEL *model, double *lin, 
		      double *a, long int *unlabeled, 
		      long int *label, long int totdoc, 
		      LEARN_PARM *learn_parm)
{
  FILE *predfl;
  long i;
  double dist,a_max;

  if(verbosity>=1) {
    printf("Writing prediction file..."); fflush(stdout);
  }
  if ((predfl = fopen (predfile, "w")) == NULL)
  { perror (predfile); exit (1); }
  a_max=learn_parm->epsilon_a;
  for(i=0;i<totdoc;i++) {
    if((unlabeled[i]) && (a[i]>a_max)) {
      a_max=a[i];
    }
  }
  for(i=0;i<totdoc;i++) {
    if(unlabeled[i]) {
      if((a[i]>(learn_parm->epsilon_a))) {
	dist=(double)label[i]*(1.0-learn_parm->epsilon_crit-a[i]/(a_max*2.0));
      }
      else {
	dist=(lin[i]-model->b);
      }
      if(dist>0) {
	fprintf(predfl,"%.8g:+1 %.8g:-1\n",dist,-dist);
      }
      else {
	fprintf(predfl,"%.8g:-1 %.8g:+1\n",-dist,dist);
      }
    }
  }
  fclose(predfl);
  if(verbosity>=1) {
    printf("done\n");
  }
}

void write_alphas(char *alphafile, double *a, 
		  long int *label, long int totdoc)
{
  FILE *alphafl;
  long i;

  if(verbosity>=1) {
    printf("Writing alpha file..."); fflush(stdout);
  }
  if ((alphafl = fopen (alphafile, "w")) == NULL)
  { perror (alphafile); exit (1); }
  for(i=0;i<totdoc;i++) {
    fprintf(alphafl,"%.18g\n",a[i]*(double)label[i]);
  }
  fclose(alphafl);
  if(verbosity>=1) {
    printf("done\n");
  }
}

