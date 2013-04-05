/***********************************************************************/
/*                                                                     */
/*   svm_hideo.c                                                       */
/*                                                                     */
/*   The Hildreth and D'Espo solver specialized for SVMs.              */
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

# include <math.h>
# include "svm_common.h"

/* 
  solve the quadratic programming problem
 
  minimize   g0 * x + 1/2 x' * G * x
  subject to ce*x = ce0
             l <= x <= u
 
  The linear constraint vector ce can only have -1/+1 as entries 
*/

/* Common Block Declarations */

long verbosity;

# define PRIMAL_OPTIMAL      1
# define DUAL_OPTIMAL        2
# define MAXITER_EXCEEDED    3
# define NAN_SOLUTION        4
# define ONLY_ONE_VARIABLE   5

# define LARGEROUND          0
# define SMALLROUND          1

/* /////////////////////////////////////////////////////////////// */

# define DEF_PRECISION          1E-5
# define DEF_MAX_ITERATIONS     200
# define DEF_LINDEP_SENSITIVITY 1E-8
# define EPSILON_HIDEO          1E-20
# define EPSILON_EQ             1E-5

double *optimize_qp(QP *, double *, long, double *, LEARN_PARM *);
double *primal=0,*dual=0;
long   precision_violations=0;
double opt_precision=DEF_PRECISION;
long   maxiter=DEF_MAX_ITERATIONS;
double lindep_sensitivity=DEF_LINDEP_SENSITIVITY;
double *buffer;
long   *nonoptimal;

long  smallroundcount=0;
long  roundnumber=0;

/* /////////////////////////////////////////////////////////////// */

void *my_malloc();

int optimize_hildreth_despo(long,long,double,double,double,long,long,long,double,double *,
			    double *,double *,double *,double *,double *,
			    double *,double *,double *,long *,double *,double *);
int solve_dual(long,long,double,double,long,double *,double *,double *,
	       double *,double *,double *,double *,double *,double *,
	       double *,double *,double *,double *,long);

void linvert_matrix(double *, long, double *, double, long *);
void lprint_matrix(double *, long);
void ladd_matrix(double *, long, double);
void lcopy_matrix(double *, long, double *);
void lswitch_rows_matrix(double *, long, long, long);
void lswitchrk_matrix(double *, long, long, long);

double calculate_qp_objective(long, double *, double *, double *);



double *optimize_qp(qp,epsilon_crit,nx,threshold,learn_parm)
QP *qp;
double *epsilon_crit;
long nx; /* Maximum number of variables in QP */
double *threshold; 
LEARN_PARM *learn_parm;
/* start the optimizer and return the optimal values */
/* The HIDEO optimizer does not necessarily fully solve the problem. */
/* Since it requires a strictly positive definite hessian, the solution */
/* is restricted to a linear independent subset in case the matrix is */
/* only semi-definite. */
{
  long i,j;
  int result;
  double eq,progress;

  roundnumber++;

  if(!primal) { /* allocate memory at first call */
    primal=(double *)my_malloc(sizeof(double)*nx);
    dual=(double *)my_malloc(sizeof(double)*((nx+1)*2));
    nonoptimal=(long *)my_malloc(sizeof(long)*(nx));
    buffer=(double *)my_malloc(sizeof(double)*((nx+1)*2*(nx+1)*2+
					       nx*nx+2*(nx+1)*2+2*nx+1+2*nx+
					       nx+nx+nx*nx));
    (*threshold)=0;
    for(i=0;i<nx;i++) {
      primal[i]=0;
    }
  }

  if(verbosity>=4) { /* really verbose */
    printf("\n\n");
    eq=qp->opt_ce0[0];
    for(i=0;i<qp->opt_n;i++) {
      eq+=qp->opt_xinit[i]*qp->opt_ce[i];
      printf("%f: ",qp->opt_g0[i]);
      for(j=0;j<qp->opt_n;j++) {
	printf("%f ",qp->opt_g[i*qp->opt_n+j]);
      }
      printf(": a=%.10f < %f",qp->opt_xinit[i],qp->opt_up[i]);
      printf(": y=%f\n",qp->opt_ce[i]);
    }
    if(qp->opt_m) {
      printf("EQ: %f*x0",qp->opt_ce[0]);
      for(i=1;i<qp->opt_n;i++) {
	printf(" + %f*x%ld",qp->opt_ce[i],i);
      }
      printf(" = %f\n\n",-qp->opt_ce0[0]);
    }
  }

  result=optimize_hildreth_despo(qp->opt_n,qp->opt_m,
				 opt_precision,(*epsilon_crit),
				 learn_parm->epsilon_a,maxiter,
				 /* (long)PRIMAL_OPTIMAL, */
				 (long)0, (long)0,
				 lindep_sensitivity,
				 qp->opt_g,qp->opt_g0,qp->opt_ce,qp->opt_ce0,
				 qp->opt_low,qp->opt_up,primal,qp->opt_xinit,
				 dual,nonoptimal,buffer,&progress);
  if(verbosity>=3) { 
    printf("return(%d)...",result);
  }

  if(learn_parm->totwords < learn_parm->svm_maxqpsize) { 
    /* larger working sets will be linear dependent anyway */
    learn_parm->svm_maxqpsize=maxl(learn_parm->totwords,(long)2);
  }

  if(result == NAN_SOLUTION) {
    lindep_sensitivity*=2;  /* throw out linear dependent examples more */
                            /* generously */
    if(learn_parm->svm_maxqpsize>2) {
      learn_parm->svm_maxqpsize--;  /* decrease size of qp-subproblems */
    }
    precision_violations++;
  }

  /* take one round of only two variable to get unstuck */
  if((result != PRIMAL_OPTIMAL) || (!(roundnumber % 31)) || (progress <= 0)) {

    smallroundcount++;

    result=optimize_hildreth_despo(qp->opt_n,qp->opt_m,
				   opt_precision,(*epsilon_crit),
				   learn_parm->epsilon_a,(long)maxiter,
				   (long)PRIMAL_OPTIMAL,(long)SMALLROUND,
				   lindep_sensitivity,
				   qp->opt_g,qp->opt_g0,qp->opt_ce,qp->opt_ce0,
				   qp->opt_low,qp->opt_up,primal,qp->opt_xinit,
				   dual,nonoptimal,buffer,&progress);
    if(verbosity>=3) { 
      printf("return_srd(%d)...",result);
    }

    if(result != PRIMAL_OPTIMAL) {
      if(result != ONLY_ONE_VARIABLE) 
	precision_violations++;
      if(result == MAXITER_EXCEEDED) 
	maxiter+=100;
      if(result == NAN_SOLUTION) {
	lindep_sensitivity*=2;  /* throw out linear dependent examples more */
	                        /* generously */
	/* results not valid, so return inital values */
	for(i=0;i<qp->opt_n;i++) {
	  primal[i]=qp->opt_xinit[i];
	}
      }
    }
  }


  if(precision_violations > 50) {
    precision_violations=0;
    (*epsilon_crit)*=10.0; 
    if(verbosity>=1) {
      printf("\nWARNING: Relaxing epsilon on KT-Conditions (%f).\n",
	     (*epsilon_crit));
    }
  }	  

  if((qp->opt_m>0) && (result != NAN_SOLUTION) && (!isnan(dual[1]-dual[0])))
    (*threshold)=dual[1]-dual[0];
  else
    (*threshold)=0;

  if(verbosity>=4) { /* really verbose */
    printf("\n\n");
    eq=qp->opt_ce0[0];
    for(i=0;i<qp->opt_n;i++) {
      eq+=primal[i]*qp->opt_ce[i];
      printf("%f: ",qp->opt_g0[i]);
      for(j=0;j<qp->opt_n;j++) {
	printf("%f ",qp->opt_g[i*qp->opt_n+j]);
      }
      printf(": a=%.30f",primal[i]);
      printf(": nonopti=%ld",nonoptimal[i]);
      printf(": y=%f\n",qp->opt_ce[i]);
    }
    printf("eq-constraint=%.30f\n",eq);
    printf("b=%f\n",(*threshold));
    printf(" smallroundcount=%ld ",smallroundcount);
  }

  return(primal);
}



int optimize_hildreth_despo(n,m,precision,epsilon_crit,epsilon_a,maxiter,goal,
			    smallround,lindep_sensitivity,g,g0,ce,ce0,low,up,
			    primal,init,dual,lin_dependent,buffer,progress)
     long   n;            /* number of variables */
     long   m;            /* number of linear equality constraints [0,1] */
     double precision;    /* solve at least to this dual precision */
     double epsilon_crit; /* stop, if KT-Conditions approx fulfilled */
     double epsilon_a;    /* precision of alphas at bounds */
     long   maxiter;      /* stop after this many iterations */
     long   goal;         /* keep going until goal fulfilled */
     long   smallround;   /* use only two variables of steepest descent */
     double lindep_sensitivity; /* epsilon for detecting linear dependent ex */
     double *g;           /* hessian of objective */
     double *g0;          /* linear part of objective */
     double *ce,*ce0;     /* linear equality constraints */
     double *low,*up;     /* box constraints */
     double *primal;      /* primal variables */
     double *init;        /* initial values of primal */
     double *dual;        /* dual variables */
     long   *lin_dependent;
     double *buffer;
     double *progress;    /* delta in the objective function between
                             before and after */
{
  long i,j,k,from,to,n_indep,changed;
  double sum,bmin=0,bmax=0;
  double *d,*d0,*ig,*dual_old,*temp,*start;       
  double *g0_new,*g_new,*ce_new,*ce0_new,*low_new,*up_new;
  double add,t;
  int result;
  double obj_before,obj_after; 
  long b1,b2;
  double g0_b1,g0_b2,ce0_b;

  g0_new=&(buffer[0]);    /* claim regions of buffer */
  d=&(buffer[n]);
  d0=&(buffer[n+(n+m)*2*(n+m)*2]);
  ce_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2]);
  ce0_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n]);
  ig=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m]);
  dual_old=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n]);
  low_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n+(n+m)*2]);
  up_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n+(n+m)*2+n]);
  start=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n+(n+m)*2+n+n]);
  g_new=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n+(n+m)*2+n+n+n]);
  temp=&(buffer[n+(n+m)*2*(n+m)*2+(n+m)*2+n+m+n*n+(n+m)*2+n+n+n+n*n]);

  b1=-1;
  b2=-1;
  for(i=0;i<n;i++) {   /* get variables with steepest feasible descent */
    sum=g0[i];         
    for(j=0;j<n;j++) 
      sum+=init[j]*g[i*n+j];
    sum=sum*ce[i];
    if(((b1==-1) || (sum<bmin)) 
       && (!((init[i]<=(low[i]+epsilon_a)) && (ce[i]<0.0)))
       && (!((init[i]>=( up[i]-epsilon_a)) && (ce[i]>0.0)))
       ) {
      bmin=sum;
      b1=i;
    }
    if(((b2==-1) || (sum>=bmax)) 
       && (!((init[i]<=(low[i]+epsilon_a)) && (ce[i]>0.0)))
       && (!((init[i]>=( up[i]-epsilon_a)) && (ce[i]<0.0)))
       ) {
      bmax=sum;
      b2=i;
    }
  }
  /* in case of unbiased hyperplane, the previous projection on */
  /* equality constraint can lead to b1 or b2 being -1. */
  if((b1 == -1) || (b2 == -1)) {
    b1=maxl(b1,b2);
    b2=maxl(b1,b2);
  }

  for(i=0;i<n;i++) {
    start[i]=init[i];
  }

  /* in case both example vectors are linearly dependent */
  /* WARNING: Assumes that ce[] in {-1,1} */
  add=0;
  changed=0;
  if((b1 != b2) && (m==1)) {
    for(i=0;i<n;i++) {  /* fix other vectors */
      if(i==b1) 
	g0_b1=g0[i];
      if(i==b2) 
	g0_b2=g0[i];
    }
    ce0_b=ce0[0];
    for(i=0;i<n;i++) {  
      if((i!=b1) && (i!=b2)) {
	for(j=0;j<n;j++) {
	  if(j==b1) 
	    g0_b1+=start[i]*g[i*n+j];
	  if(j==b2) 
	    g0_b2+=start[i]*g[i*n+j];
	}
	ce0_b-=(start[i]*ce[i]);
      }
    }
    if((g[b1*n+b2] == g[b1*n+b1]) && (g[b1*n+b2] == g[b2*n+b2])) {
      /* printf("euqal\n"); */
      if(ce[b1] == ce[b2]) { 
	if(g0_b1 <= g0_b2) { /* set b1 to upper bound */
	  /* printf("case +=<\n"); */
	  changed=1;
	  t=up[b1]-init[b1];
	  if((init[b2]-low[b2]) < t) {
	    t=init[b2]-low[b2];
	  }
	  start[b1]=init[b1]+t;
	  start[b2]=init[b2]-t;
	}
	else if(g0_b1 > g0_b2) { /* set b2 to upper bound */
	  /* printf("case +=>\n"); */
	  changed=1;
	  t=up[b2]-init[b2];
	  if((init[b1]-low[b1]) < t) {
	    t=init[b1]-low[b1];
	  }
	  start[b1]=init[b1]-t;
	  start[b2]=init[b2]+t;
	}
      }
      else if(((g[b1*n+b1]>0) || (g[b2*n+b2]>0))) { /* (ce[b1] != ce[b2]) */ 
	/* printf("case +!\n"); */
	t=((ce[b2]/ce[b1])*g0[b1]-g0[b2]+ce0[0]*(g[b1*n+b1]*ce[b2]/ce[b1]-g[b1*n+b2]/ce[b1]))/((ce[b2]*ce[b2]/(ce[b1]*ce[b1]))*g[b1*n+b1]+g[b2*n+b2]-2*(g[b1*n+b2]*ce[b2]/ce[b1]))-init[b2];
	changed=1;
	if((up[b2]-init[b2]) < t) {
	  t=up[b2]-init[b2];
	}
	if((init[b2]-low[b2]) < -t) {
	  t=-(init[b2]-low[b2]);
	}
	if((up[b1]-init[b1]) < t) {
	  t=(up[b1]-init[b1]);
	}
	if((init[b1]-low[b1]) < -t) {
	  t=-(init[b1]-low[b1]);
	}
	start[b1]=init[b1]+t;
	start[b2]=init[b2]+t;
      }
    }
    if((-g[b1*n+b2] == g[b1*n+b1]) && (-g[b1*n+b2] == g[b2*n+b2])) {
      /* printf("diffeuqal\n"); */
      if(ce[b1] != ce[b2]) {
	if((g0_b1+g0_b2) < 0) { /* set b1 and b2 to upper bound */
	  /* printf("case -!<\n"); */
	  changed=1;
	  t=up[b1]-init[b1];
	  if((up[b2]-init[b2]) < t) {
	    t=up[b2]-init[b2];
	  }
	  start[b1]=init[b1]+t;
	  start[b2]=init[b2]+t;
	}     
	else if((g0_b1+g0_b2) >= 0) { /* set b1 and b2 to lower bound */
	  /* printf("case -!>\n"); */
	  changed=1;
	  t=init[b1]-low[b1];
	  if((init[b2]-low[b2]) < t) {
	    t=init[b2]-low[b2];
	  }
	  start[b1]=init[b1]-t;
	  start[b2]=init[b2]-t;
	}
      }
      else if(((g[b1*n+b1]>0) || (g[b2*n+b2]>0))) { /* (ce[b1]==ce[b2]) */
	/*  printf("case -=\n"); */
	t=((ce[b2]/ce[b1])*g0[b1]-g0[b2]+ce0[0]*(g[b1*n+b1]*ce[b2]/ce[b1]-g[b1*n+b2]/ce[b1]))/((ce[b2]*ce[b2]/(ce[b1]*ce[b1]))*g[b1*n+b1]+g[b2*n+b2]-2*(g[b1*n+b2]*ce[b2]/ce[b1]))-init[b2];
	changed=1;
	if((up[b2]-init[b2]) < t) {
	  t=up[b2]-init[b2];
	}
	if((init[b2]-low[b2]) < -t) {
	  t=-(init[b2]-low[b2]);
	}
	if((up[b1]-init[b1]) < -t) {
	  t=-(up[b1]-init[b1]);
	}
	if((init[b1]-low[b1]) < t) {
	  t=init[b1]-low[b1];
	}
	start[b1]=init[b1]-t;
	start[b2]=init[b2]+t;
      }	
    }
  }
  /* if we have a biased hyperplane, then adding a constant to the */
  /* hessian does not change the solution. So that is done for examples */
  /* with zero diagonal entry, since HIDEO cannot handle them. */
  if((m>0) 
     && ((fabs(g[b1*n+b1]) < lindep_sensitivity) 
	 || (fabs(g[b2*n+b2]) < lindep_sensitivity))) {
    /* printf("Case 0\n"); */
    add+=0.093274;
  }    
  /* in case both examples are linear dependent */
  else if((m>0) 
	  && (g[b1*n+b2] != 0 && g[b2*n+b2] != 0)
	  && (fabs(g[b1*n+b1]/g[b1*n+b2] - g[b1*n+b2]/g[b2*n+b2])
	      < lindep_sensitivity)) { 
    /* printf("Case lindep\n"); */
    add+=0.078274;
  }

  /* special case for zero diagonal entry on unbiased hyperplane */
  if((m==0) && (b1>=0))  {
    if(fabs(g[b1*n+b1]) < lindep_sensitivity) { 
      /* printf("Case 0b1\n"); */
      for(i=0;i<n;i++) {  /* fix other vectors */
	if(i==b1) 
	  g0_b1=g0[i];
      }
      for(i=0;i<n;i++) {  
	if(i!=b1) {
	  for(j=0;j<n;j++) {
	    if(j==b1) 
	      g0_b1+=start[i]*g[i*n+j];
	  }
	}
      }
      if(g0_b1<0)
	start[b1]=up[b1];
      if(g0_b1>=0)
	start[b1]=low[b1];
    }
  }
  if((m==0) && (b2>=0))  {
    if(fabs(g[b2*n+b2]) < lindep_sensitivity) { 
      /* printf("Case 0b2\n"); */
      for(i=0;i<n;i++) {  /* fix other vectors */
	if(i==b2) 
	  g0_b2=g0[i];
      }
      for(i=0;i<n;i++) {  
	if(i!=b2) {
	  for(j=0;j<n;j++) {
	    if(j==b2) 
	      g0_b2+=start[i]*g[i*n+j];
	  }
	}
      }
      if(g0_b2<0)
	start[b2]=up[b2];
      if(g0_b2>=0)
	start[b2]=low[b2];
    }
  }

  /* printf("b1=%ld,b2=%ld\n",b1,b2); */

  lcopy_matrix(g,n,d);
  if((m==1) && (add>0.0)) {
    for(j=0;j<n;j++) {
      for(k=0;k<n;k++) {
	d[j*n+k]+=add*ce[j]*ce[k];
      }
    }
  }
  else {
    add=0.0;
  }

  if(n>2) {                    /* switch, so that variables are better mixed */
    lswitchrk_matrix(d,n,b1,(long)0); 
    if(b2 == 0) 
      lswitchrk_matrix(d,n,b1,(long)1); 
    else
      lswitchrk_matrix(d,n,b2,(long)1); 
  }
  if(smallround == SMALLROUND) {
    for(i=2;i<n;i++) {
      lin_dependent[i]=1;
    }
    if(m>0) { /* for biased hyperplane, pick two variables */
      lin_dependent[0]=0;
      lin_dependent[1]=0;
    }
    else {    /* for unbiased hyperplane, pick only one variable */
      lin_dependent[0]=smallroundcount % 2;
      lin_dependent[1]=(smallroundcount+1) % 2;
    }
  }
  else {
    for(i=0;i<n;i++) {
      lin_dependent[i]=0;
    }
  }
  linvert_matrix(d,n,ig,lindep_sensitivity,lin_dependent);
  if(n>2) {                    /* now switch back */
    if(b2 == 0) {
      lswitchrk_matrix(ig,n,b1,(long)1); 
      i=lin_dependent[1];  
      lin_dependent[1]=lin_dependent[b1];
      lin_dependent[b1]=i;
    }
    else {
      lswitchrk_matrix(ig,n,b2,(long)1); 
      i=lin_dependent[1];  
      lin_dependent[1]=lin_dependent[b2];
      lin_dependent[b2]=i;
    }
    lswitchrk_matrix(ig,n,b1,(long)0); 
    i=lin_dependent[0];  
    lin_dependent[0]=lin_dependent[b1];
    lin_dependent[b1]=i;
  }
  /* lprint_matrix(d,n); */
  /* lprint_matrix(ig,n); */

  lcopy_matrix(g,n,g_new);   /* restore g_new matrix */
  if(add>0)
    for(j=0;j<n;j++) {
      for(k=0;k<n;k++) {
	g_new[j*n+k]+=add*ce[j]*ce[k];
      }
    }

  for(i=0;i<n;i++) {  /* fix linear dependent vectors */
    g0_new[i]=g0[i]+add*ce0[0]*ce[i];
  }
  if(m>0) ce0_new[0]=-ce0[0];
  for(i=0;i<n;i++) {  /* fix linear dependent vectors */
    if(lin_dependent[i]) {
      for(j=0;j<n;j++) {
	if(!lin_dependent[j]) {
	  g0_new[j]+=start[i]*g_new[i*n+j];
	}
      }
      if(m>0) ce0_new[0]-=(start[i]*ce[i]);
    }
  }
  from=0;   /* remove linear dependent vectors */
  to=0;
  n_indep=0;
  for(i=0;i<n;i++) {
    if(!lin_dependent[i]) {
      g0_new[n_indep]=g0_new[i];
      ce_new[n_indep]=ce[i]; 
      low_new[n_indep]=low[i];
      up_new[n_indep]=up[i];
      primal[n_indep]=start[i];
      n_indep++;
    }
    for(j=0;j<n;j++) {
      if((!lin_dependent[i]) && (!lin_dependent[j])) {
        ig[to]=ig[from];
        g_new[to]=g_new[from];
	to++;
      }
      from++;
    }
  }

  if(verbosity>=3) {
    printf("real_qp_size(%ld)...",n_indep);
  }
  
  /* cannot optimize with only one variable */
  if((n_indep<=1) && (m>0) && (!changed)) { 
    for(i=n-1;i>=0;i--) {
      primal[i]=init[i];
    }
    return((int)ONLY_ONE_VARIABLE);
  }

  if((!changed) || (n_indep>1)) { 
    result=solve_dual(n_indep,m,precision,epsilon_crit,maxiter,g_new,g0_new,
		      ce_new,ce0_new,low_new,up_new,primal,d,d0,ig,
		      dual,dual_old,temp,goal);
  }
  else {
    result=PRIMAL_OPTIMAL;
  }
  
  j=n_indep;
  for(i=n-1;i>=0;i--) {
    if(!lin_dependent[i]) {
      j--;
      primal[i]=primal[j];
    }
    else {
      primal[i]=start[i];  /* leave as is */
    }
    temp[i]=primal[i];
  }
   
  obj_before=calculate_qp_objective(n,g,g0,init);
  obj_after=calculate_qp_objective(n,g,g0,primal);
  (*progress)=obj_before-obj_after;
  if(verbosity>=3) {
    printf("before(%.30f)...after(%.30f)...result_sd(%d)...",
	   obj_before,obj_after,result); 
  }

  return((int)result);
}


int solve_dual(n,m,precision,epsilon_crit,maxiter,g,g0,ce,ce0,low,up,primal,
	       d,d0,ig,dual,dual_old,temp,goal)
     /* Solves the dual using the method of Hildreth and D'Espo. */
     /* Can only handle problems with zero or exactly one */
     /* equality constraints. */

     long   n;            /* number of variables */
     long   m;            /* number of linear equality constraints */
     double precision;    /* solve at least to this dual precision */
     double epsilon_crit; /* stop, if KT-Conditions approx fulfilled */
     long   maxiter;      /* stop after that many iterations */
     double *g;
     double *g0;          /* linear part of objective */
     double *ce,*ce0;     /* linear equality constraints */
     double *low,*up;     /* box constraints */
     double *primal;      /* variables (with initial values) */
     double *d,*d0,*ig,*dual,*dual_old,*temp;       /* buffer  */
     long goal;
{
  long i,j,k,iter;
  double sum,w,maxviol,viol,temp1,temp2,isnantest;
  double model_b,dist;
  long retrain,maxfaktor,primal_optimal=0,at_bound,scalemaxiter;
  double epsilon_a=1E-15,epsilon_hideo;
  double eq; 

  if((m<0) || (m>1)) 
    perror("SOLVE DUAL: inappropriate number of eq-constrains!");

  /*  
  printf("\n");
  for(i=0;i<n;i++) {
    printf("%f: ",g0[i]);
    for(j=0;j<n;j++) {
      printf("%f ",g[i*n+j]);
    }
    printf(": a=%.30f",primal[i]);
    printf(": y=%f\n",ce[i]);
  }
  */

  for(i=0;i<2*(n+m);i++) {
    dual[i]=0;
    dual_old[i]=0;
  }
  for(i=0;i<n;i++) {   
    for(j=0;j<n;j++) {   /* dual hessian for box constraints */
      d[i*2*(n+m)+j]=ig[i*n+j];
      d[(i+n)*2*(n+m)+j]=-ig[i*n+j];
      d[i*2*(n+m)+j+n]=-ig[i*n+j];
      d[(i+n)*2*(n+m)+j+n]=ig[i*n+j];
    }
    if(m>0) {
      sum=0;              /* dual hessian for eq constraints */
      for(j=0;j<n;j++) {
	sum+=(ce[j]*ig[i*n+j]);
      }
      d[i*2*(n+m)+2*n]=sum;
      d[i*2*(n+m)+2*n+1]=-sum;
      d[(n+i)*2*(n+m)+2*n]=-sum;
      d[(n+i)*2*(n+m)+2*n+1]=sum;
      d[(n+n)*2*(n+m)+i]=sum;
      d[(n+n+1)*2*(n+m)+i]=-sum;
      d[(n+n)*2*(n+m)+(n+i)]=-sum;
      d[(n+n+1)*2*(n+m)+(n+i)]=sum;
      
      sum=0;
      for(j=0;j<n;j++) {
	for(k=0;k<n;k++) {
	  sum+=(ce[k]*ce[j]*ig[j*n+k]);
	}
      }
      d[(n+n)*2*(n+m)+2*n]=sum;
      d[(n+n)*2*(n+m)+2*n+1]=-sum;
      d[(n+n+1)*2*(n+m)+2*n]=-sum;
      d[(n+n+1)*2*(n+m)+2*n+1]=sum;
    } 
  }

  for(i=0;i<n;i++) {   /* dual linear component for the box constraints */
    w=0;
    for(j=0;j<n;j++) {
      w+=(ig[i*n+j]*g0[j]); 
    }
    d0[i]=up[i]+w;
    d0[i+n]=-low[i]-w;
  }

  if(m>0) {  
    sum=0;             /* dual linear component for eq constraints */
    for(j=0;j<n;j++) {
      for(k=0;k<n;k++) {
	sum+=(ce[k]*ig[k*n+j]*g0[j]); 
      }
    }
    d0[2*n]=ce0[0]+sum;
    d0[2*n+1]=-ce0[0]-sum;
  }

  maxviol=999999;
  iter=0;
  retrain=1;
  maxfaktor=1;
  scalemaxiter=maxiter/5;
  while((retrain) && (maxviol > 0) && (iter < (scalemaxiter*maxfaktor))) {
    iter++;
    
    while((maxviol > precision) && (iter < (scalemaxiter*maxfaktor))) {
      iter++;
      maxviol=0;
      for(i=0;i<2*(n+m);i++) {
	sum=d0[i];
	for(j=0;j<2*(n+m);j++) {
	  sum+=d[i*2*(n+m)+j]*dual_old[j];
	}
	sum-=d[i*2*(n+m)+i]*dual_old[i];
	dual[i]=-sum/d[i*2*(n+m)+i];
	if(dual[i]<0) dual[i]=0;
	
	viol=fabs(dual[i]-dual_old[i]);
	if(viol>maxviol) 
	  maxviol=viol;
	dual_old[i]=dual[i];
      }
      /*
      printf("%d) maxviol=%20f precision=%f\n",iter,maxviol,precision); 
      */
    }
  
    if(m>0) {
      for(i=0;i<n;i++) {
	temp[i]=dual[i]-dual[i+n]+ce[i]*(dual[n+n]-dual[n+n+1])+g0[i];
      }
    } 
    else {
      for(i=0;i<n;i++) {
	temp[i]=dual[i]-dual[i+n]+g0[i];
      }
    }
    for(i=0;i<n;i++) {
      primal[i]=0;             /* calc value of primal variables */
      for(j=0;j<n;j++) {
	primal[i]+=ig[i*n+j]*temp[j];
      }
      primal[i]*=-1.0;
      if(primal[i]<=(low[i])) {  /* clip conservatively */
	primal[i]=low[i];
      }
      else if(primal[i]>=(up[i])) {
	primal[i]=up[i];
      }
    }

    if(m>0) 
      model_b=dual[n+n+1]-dual[n+n];
    else
      model_b=0;

    epsilon_hideo=EPSILON_HIDEO;
    for(i=0;i<n;i++) {           /* check precision of alphas */
      dist=-model_b*ce[i]; 
      dist+=(g0[i]+1.0);
      for(j=0;j<i;j++) {
	dist+=(primal[j]*g[j*n+i]);
      }
      for(j=i;j<n;j++) {
	dist+=(primal[j]*g[i*n+j]);
      }
      if((primal[i]<(up[i]-epsilon_hideo)) && (dist < (1.0-epsilon_crit))) {
	epsilon_hideo=(up[i]-primal[i])*2.0;
      }
      else if((primal[i]>(low[i]+epsilon_hideo)) &&(dist>(1.0+epsilon_crit))) {
	epsilon_hideo=(primal[i]-low[i])*2.0;
      }
    }
    /* printf("\nEPSILON_HIDEO=%.30f\n",epsilon_hideo); */

    for(i=0;i<n;i++) {           /* clip alphas to bounds */
      if(primal[i]<=(low[i]+epsilon_hideo)) {
	primal[i]=low[i];
      }
      else if(primal[i]>=(up[i]-epsilon_hideo)) {
	primal[i]=up[i];
      }
    }

    retrain=0;
    primal_optimal=1;
    at_bound=0;
    for(i=0;(i<n);i++) {  /* check primal KT-Conditions */
      dist=-model_b*ce[i]; 
      dist+=(g0[i]+1.0);
      for(j=0;j<i;j++) {
	dist+=(primal[j]*g[j*n+i]);
      }
      for(j=i;j<n;j++) {
	dist+=(primal[j]*g[i*n+j]);
      }
      if((primal[i]<(up[i]-epsilon_a)) && (dist < (1.0-epsilon_crit))) {
	retrain=1;
	primal_optimal=0;
      }
      else if((primal[i]>(low[i]+epsilon_a)) && (dist > (1.0+epsilon_crit))) {
	retrain=1;
	primal_optimal=0;
      }
      if((primal[i]<=(low[i]+epsilon_a)) || (primal[i]>=(up[i]-epsilon_a))) {
	at_bound++;
      }
      /*    printf("HIDEOtemp: a[%ld]=%.30f, dist=%.6f, b=%f, at_bound=%ld\n",i,primal[i],dist,model_b,at_bound);  */
    }
    if(m>0) {
      eq=-ce0[0];               /* check precision of eq-constraint */
      for(i=0;i<n;i++) { 
	eq+=(ce[i]*primal[i]);
      }
      if((EPSILON_EQ < fabs(eq)) 
	 /*
	 && !((goal==PRIMAL_OPTIMAL) 
	       && (at_bound==n)) */
	 ) {
	retrain=1;
	primal_optimal=0;
      }
      /* printf("\n eq=%.30f ce0=%f at-bound=%ld\n",eq,ce0[0],at_bound);  */
    }

    if(retrain) {
      precision/=10;
      if(((goal == PRIMAL_OPTIMAL) && (maxfaktor < 50000))
	 || (maxfaktor < 5)) {
	maxfaktor++;
      }
    }
  }

  if(!primal_optimal) {
    for(i=0;i<n;i++) {
      primal[i]=0;             /* calc value of primal variables */
      for(j=0;j<n;j++) {
	primal[i]+=ig[i*n+j]*temp[j];
      }
      primal[i]*=-1.0;
      if(primal[i]<=(low[i]+epsilon_a)) {  /* clip conservatively */
	primal[i]=low[i];
      }
      else if(primal[i]>=(up[i]-epsilon_a)) {
	primal[i]=up[i];
      }
    }
  }

  isnantest=0;
  for(i=0;i<n;i++) {           /* check for isnan */
    isnantest+=primal[i];
  }

  if(m>0) {
    temp1=dual[n+n+1];   /* copy the dual variables for the eq */
    temp2=dual[n+n];     /* constraints to a handier location */
    for(i=n+n+1;i>=2;i--) {
      dual[i]=dual[i-2];
    }
    dual[0]=temp2;
    dual[1]=temp1;
    isnantest+=temp1+temp2;
  }

  if(isnan(isnantest)) {
    return((int)NAN_SOLUTION);
  }
  else if(primal_optimal) {
    return((int)PRIMAL_OPTIMAL);
  }
  else if(maxviol == 0.0) {
    return((int)DUAL_OPTIMAL);
  }
  else {
    return((int)MAXITER_EXCEEDED);
  }
}


void linvert_matrix(matrix,depth,inverse,lindep_sensitivity,lin_dependent)
double *matrix;
long depth;
double *inverse,lindep_sensitivity;
long *lin_dependent;  /* indicates the active parts of matrix on 
			 input and output*/
{
  long i,j,k;
  double factor;

  for(i=0;i<depth;i++) {
    /*    lin_dependent[i]=0; */
    for(j=0;j<depth;j++) {
      inverse[i*depth+j]=0.0;
    }
    inverse[i*depth+i]=1.0;
  }
  for(i=0;i<depth;i++) {
    if(lin_dependent[i] || (fabs(matrix[i*depth+i])<lindep_sensitivity)) {
      lin_dependent[i]=1;
    }
    else {
      for(j=i+1;j<depth;j++) {
	factor=matrix[j*depth+i]/matrix[i*depth+i];
	for(k=i;k<depth;k++) {
	  matrix[j*depth+k]-=(factor*matrix[i*depth+k]);
	}
	for(k=0;k<depth;k++) {
	  inverse[j*depth+k]-=(factor*inverse[i*depth+k]);
	}
      }
    }
  }
  for(i=depth-1;i>=0;i--) {
    if(!lin_dependent[i]) {
      factor=1/matrix[i*depth+i];
      for(k=0;k<depth;k++) {
	inverse[i*depth+k]*=factor;
      }
      matrix[i*depth+i]=1;
      for(j=i-1;j>=0;j--) {
	factor=matrix[j*depth+i];
	matrix[j*depth+i]=0;
	for(k=0;k<depth;k++) {
	  inverse[j*depth+k]-=(factor*inverse[i*depth+k]);
	}
      }
    }
  }
}

void lprint_matrix(matrix,depth)
double *matrix;
long depth;
{
  long i,j;
  for(i=0;i<depth;i++) {
    for(j=0;j<depth;j++) {
      printf("%5.2f ",(double)(matrix[i*depth+j]));
    }
    printf("\n");
  }
  printf("\n");
}

void ladd_matrix(matrix,depth,scalar)
double *matrix;
long depth;
double scalar;
{
  long i,j;
  for(i=0;i<depth;i++) {
    for(j=0;j<depth;j++) {
      matrix[i*depth+j]+=scalar;
    }
  }
}

void lcopy_matrix(matrix,depth,matrix2) 
double *matrix;
long depth;
double *matrix2;
{
  long i;
  
  for(i=0;i<(depth)*(depth);i++) {
    matrix2[i]=matrix[i];
  }
}

void lswitch_rows_matrix(matrix,depth,r1,r2) 
double *matrix;
long depth,r1,r2;
{
  long i;
  double temp;

  for(i=0;i<depth;i++) {
    temp=matrix[r1*depth+i];
    matrix[r1*depth+i]=matrix[r2*depth+i];
    matrix[r2*depth+i]=temp;
  }
}

void lswitchrk_matrix(matrix,depth,rk1,rk2) 
double *matrix;
long depth,rk1,rk2;
{
  long i;
  double temp;

  for(i=0;i<depth;i++) {
    temp=matrix[rk1*depth+i];
    matrix[rk1*depth+i]=matrix[rk2*depth+i];
    matrix[rk2*depth+i]=temp;
  }
  for(i=0;i<depth;i++) {
    temp=matrix[i*depth+rk1];
    matrix[i*depth+rk1]=matrix[i*depth+rk2];
    matrix[i*depth+rk2]=temp;
  }
}

double calculate_qp_objective(opt_n,opt_g,opt_g0,alpha)
long opt_n;
double *opt_g,*opt_g0,*alpha;
{
  double obj;
  long i,j;
  obj=0;  /* calculate objective  */
  for(i=0;i<opt_n;i++) {
    obj+=(opt_g0[i]*alpha[i]);
    obj+=(0.5*alpha[i]*alpha[i]*opt_g[i*opt_n+i]);
    for(j=0;j<i;j++) {
      obj+=(alpha[j]*alpha[i]*opt_g[j*opt_n+i]);
    }
  }
  return(obj);
}
