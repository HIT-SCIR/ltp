/************************************************************************/
/*                                                                      */
/*   svm_common.c                                                       */
/*                                                                      */
/*   Definitions and functions used in both svm_learn and svm_classify. */
/*                                                                      */
/*   Author: Thorsten Joachims                                          */
/*   Date: 02.07.04                                                     */
/*                                                                      */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved        */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

# include "ctype.h"
# include "svm_common.h"
# include "kernel.h"           /* this contains a user supplied kernel */

long   verbosity;              /* verbosity level (0-4) */
long   kernel_cache_statistic;

double classify_example(MODEL *model, DOC *ex) 
     /* classifies one example */
{
  register long i;
  register double dist;

  if((model->kernel_parm.kernel_type == LINEAR) && (model->lin_weights))
    return(classify_example_linear(model,ex));
	   
  dist=0;
  for(i=1;i<model->sv_num;i++) {  
    dist+=kernel(&model->kernel_parm,model->supvec[i],ex)*model->alpha[i];
  }
  return(dist-model->b);
}

double classify_example_linear(MODEL *model, DOC *ex) 
     /* classifies example for linear kernel */
     
     /* important: the model must have the linear weight vector computed */
     /* use: add_weight_vector_to_linear_model(&model); */


     /* important: the feature numbers in the example to classify must */
     /*            not be larger than the weight vector!               */
{
  double sum=0;
  SVECTOR *f;

  for(f=ex->fvec;f;f=f->next)  
    sum+=f->factor*sprod_ns(model->lin_weights,f);
  return(sum-model->b);
}


double kernel(KERNEL_PARM *kernel_parm, DOC *a, DOC *b) 
     /* calculate the kernel function */
{
  double sum=0;
  SVECTOR *fa,*fb;

  /* in case the constraints are sums of feature vector as represented
     as a list of SVECTOR's with their coefficient factor in the sum,
     take the kernel between all pairs */ 
  for(fa=a->fvec;fa;fa=fa->next) { 
    for(fb=b->fvec;fb;fb=fb->next) {
      if(fa->kernel_id == fb->kernel_id)
	sum+=fa->factor*fb->factor*single_kernel(kernel_parm,fa,fb);
    }
  }
  return(sum);
}

double single_kernel(KERNEL_PARM *kernel_parm, SVECTOR *a, SVECTOR *b) 
     /* calculate the kernel function between two vectors */
{
  kernel_cache_statistic++;
  switch(kernel_parm->kernel_type) {
    case 0: /* linear */ 
            return(sprod_ss(a,b)); 
    case 1: /* polynomial */
            return(pow(kernel_parm->coef_lin*sprod_ss(a,b)+kernel_parm->coef_const,(double)kernel_parm->poly_degree)); 
    case 2: /* radial basis function */
            return(exp(-kernel_parm->rbf_gamma*(a->twonorm_sq-2*sprod_ss(a,b)+b->twonorm_sq)));
    case 3: /* sigmoid neural net */
            return(tanh(kernel_parm->coef_lin*sprod_ss(a,b)+kernel_parm->coef_const)); 
    case 4: /* custom-kernel supplied in file kernel.h*/
            return(custom_kernel(kernel_parm,a,b)); 
    default: printf("Error: Unknown kernel function\n"); exit(1);
  }
}


SVECTOR *create_svector(WORD *words,char *userdefined,double factor)
{
  SVECTOR *vec;
  long    fnum,i;

  fnum=0;
  while(words[fnum].wnum) {
    fnum++;
  }
  fnum++;
  vec = (SVECTOR *)my_malloc(sizeof(SVECTOR));
  vec->words = (WORD *)my_malloc(sizeof(WORD)*(fnum));
  for(i=0;i<fnum;i++) { 
      vec->words[i]=words[i];
  }
  vec->twonorm_sq=sprod_ss(vec,vec);

  fnum=0;
  while(userdefined[fnum]) {
    fnum++;
  }
  fnum++;
  vec->userdefined = (char *)my_malloc(sizeof(char)*(fnum));
  for(i=0;i<fnum;i++) { 
      vec->userdefined[i]=userdefined[i];
  }
  vec->kernel_id=0;
  vec->next=NULL;
  vec->factor=factor;
  return(vec);
}

SVECTOR *copy_svector(SVECTOR *vec)
{
  SVECTOR *newvec=NULL;
  if(vec) {
    newvec=create_svector(vec->words,vec->userdefined,vec->factor);
    newvec->next=copy_svector(vec->next);
  }
  return(newvec);
}
    
void free_svector(SVECTOR *vec)
{
  if(vec) {
    free(vec->words);
    if(vec->userdefined)
      free(vec->userdefined);
    free_svector(vec->next);
    free(vec);
  }
}

double sprod_ss(SVECTOR *a, SVECTOR *b) 
     /* compute the inner product of two sparse vectors */
{
    register double sum=0;
    register WORD *ai,*bj;
    ai=a->words;
    bj=b->words;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	ai++;
      }
      else {
	sum+=(ai->weight) * (bj->weight);
	ai++;
	bj++;
      }
    }
    return((double)sum);
}

SVECTOR* sub_ss(SVECTOR *a, SVECTOR *b) 
     /* compute the difference a-b of two sparse vectors */
     /* Note: SVECTOR lists are not followed, but only the first
	SVECTOR is used */
{
    SVECTOR *vec;
    register WORD *sum,*sumi;
    register WORD *ai,*bj;
    long veclength;
  
    ai=a->words;
    bj=b->words;
    veclength=0;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	veclength++;
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	veclength++;
	ai++;
      }
      else {
	veclength++;
	ai++;
	bj++;
      }
    }
    while (bj->wnum) {
      veclength++;
      bj++;
    }
    while (ai->wnum) {
      veclength++;
      ai++;
    }
    veclength++;

    sum=(WORD *)my_malloc(sizeof(WORD)*veclength);
    sumi=sum;
    ai=a->words;
    bj=b->words;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	(*sumi)=(*bj);
	sumi->weight*=(-1);
	sumi++;
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	(*sumi)=(*ai);
	sumi++;
	ai++;
      }
      else {
	(*sumi)=(*ai);
	sumi->weight-=bj->weight;
	if(sumi->weight != 0)
	  sumi++;
	ai++;
	bj++;
      }
    }
    while (bj->wnum) {
      (*sumi)=(*bj);
      sumi->weight*=(-1);
      sumi++;
      bj++;
    }
    while (ai->wnum) {
      (*sumi)=(*ai);
      sumi++;
      ai++;
    }
    sumi->wnum=0;

    vec=create_svector(sum,"",1.0);
    free(sum);

    return(vec);
}

SVECTOR* add_ss(SVECTOR *a, SVECTOR *b) 
     /* compute the sum a+b of two sparse vectors */
     /* Note: SVECTOR lists are not followed, but only the first
	SVECTOR is used */
{
    SVECTOR *vec;
    register WORD *sum,*sumi;
    register WORD *ai,*bj;
    long veclength;
  
    ai=a->words;
    bj=b->words;
    veclength=0;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	veclength++;
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	veclength++;
	ai++;
      }
      else {
	veclength++;
	ai++;
	bj++;
      }
    }
    while (bj->wnum) {
      veclength++;
      bj++;
    }
    while (ai->wnum) {
      veclength++;
      ai++;
    }
    veclength++;

    /*** is veclength=lengSequence(a)+lengthSequence(b)? ***/

    sum=(WORD *)my_malloc(sizeof(WORD)*veclength);
    sumi=sum;
    ai=a->words;
    bj=b->words;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	(*sumi)=(*bj);
	sumi++;
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	(*sumi)=(*ai);
	sumi++;
	ai++;
      }
      else {
	(*sumi)=(*ai);
	sumi->weight+=bj->weight;
	if(sumi->weight != 0)
	  sumi++;
	ai++;
	bj++;
      }
    }
    while (bj->wnum) {
      (*sumi)=(*bj);
      sumi++;
      bj++;
    }
    while (ai->wnum) {
      (*sumi)=(*ai);
      sumi++;
      ai++;
    }
    sumi->wnum=0;

    vec=create_svector(sum,"",1.0);
    free(sum);

    return(vec);
}

SVECTOR* add_list_ss(SVECTOR *a) 
     /* computes the linear combination of the SVECTOR list weighted
	by the factor of each SVECTOR */
{
  SVECTOR *scaled,*oldsum,*sum,*f;
  WORD    empty[2];
    
  if(a){
    sum=smult_s(a,a->factor);
    for(f=a->next;f;f=f->next) {
      scaled=smult_s(f,f->factor);
      oldsum=sum;
      sum=add_ss(sum,scaled);
      free_svector(oldsum);
      free_svector(scaled);
    }
    sum->factor=1.0;
  }
  else {
    empty[0].wnum=0;
    sum=create_svector(empty,"",1.0);
  }
  return(sum);
}

void append_svector_list(SVECTOR *a, SVECTOR *b) 
     /* appends SVECTOR b to the end of SVECTOR a. */
{
    SVECTOR *f;
    
    for(f=a;f->next;f=f->next);  /* find end of first vector list */
    f->next=b;                   /* append the two vector lists */
}

SVECTOR* smult_s(SVECTOR *a, double factor) 
     /* scale sparse vector a by factor */
{
    SVECTOR *vec;
    register WORD *sum,*sumi;
    register WORD *ai;
    long veclength;
  
    ai=a->words;
    veclength=0;
    while (ai->wnum) {
      veclength++;
      ai++;
    }
    veclength++;

    sum=(WORD *)my_malloc(sizeof(WORD)*veclength);
    sumi=sum;
    ai=a->words;
    while (ai->wnum) {
	(*sumi)=(*ai);
	sumi->weight*=factor;
	if(sumi->weight != 0)
	  sumi++;
	ai++;
    }
    sumi->wnum=0;

    vec=create_svector(sum,a->userdefined,a->factor);
    free(sum);

    return(vec);
}

int featvec_eq(SVECTOR *a, SVECTOR *b)
     /* tests two sparse vectors for equality */
{
    register WORD *ai,*bj;
    ai=a->words;
    bj=b->words;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	if((bj->weight) != 0)
	  return(0);
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	if((ai->weight) != 0)
	  return(0);
	ai++;
      }
      else {
	if((ai->weight) != (bj->weight)) 
	  return(0);
	ai++;
	bj++;
      }
    }
    return(1);
}

double model_length_s(MODEL *model, KERNEL_PARM *kernel_parm) 
     /* compute length of weight vector */
{
  register long i,j;
  register double sum=0,alphai;
  register DOC *supveci;

  for(i=1;i<model->sv_num;i++) {  
    alphai=model->alpha[i];
    supveci=model->supvec[i];
    for(j=1;j<model->sv_num;j++) {
      sum+=alphai*model->alpha[j]
	   *kernel(kernel_parm,supveci,model->supvec[j]);
    }
  }
  return(sqrt(sum));
}

void clear_vector_n(double *vec, long int n)
{
  register long i;
  for(i=0;i<=n;i++) vec[i]=0;
}

void add_vector_ns(double *vec_n, SVECTOR *vec_s, double faktor)
{
  register WORD *ai;
  ai=vec_s->words;
  while (ai->wnum) {
    vec_n[ai->wnum]+=(faktor*ai->weight);
    ai++;
  }
}

double sprod_ns(double *vec_n, SVECTOR *vec_s)
{
  register double sum=0;
  register WORD *ai;
  ai=vec_s->words;
  while (ai->wnum) {
    sum+=(vec_n[ai->wnum]*ai->weight);
    ai++;
  }
  return(sum);
}

void add_weight_vector_to_linear_model(MODEL *model)
     /* compute weight vector in linear case and add to model */
{
  long i;
  SVECTOR *f;

  model->lin_weights=(double *)my_malloc(sizeof(double)*(model->totwords+1));
  clear_vector_n(model->lin_weights,model->totwords);
  for(i=1;i<model->sv_num;i++) {
    for(f=(model->supvec[i])->fvec;f;f=f->next)  
      add_vector_ns(model->lin_weights,f,f->factor*model->alpha[i]);
  }
}


DOC *create_example(long docnum, long queryid, long slackid, 
		    double costfactor, SVECTOR *fvec)
{
  DOC *example;
  example = (DOC *)my_malloc(sizeof(DOC));
  example->docnum=docnum;
  example->queryid=queryid;
  example->slackid=slackid;
  example->costfactor=costfactor;
  example->fvec=fvec;
  return(example);
}

void free_example(DOC *example, long deep)
{
  if(example) {
    if(deep) {
      if(example->fvec)
	free_svector(example->fvec);
    }
    free(example);
  }
}

void write_model(char *modelfile, MODEL *model)
{
  FILE *modelfl;
  long j,i,sv_num;
  SVECTOR *v;

  if(verbosity>=1) {
    printf("Writing model file..."); fflush(stdout);
  }
  if ((modelfl = fopen (modelfile, "w")) == NULL)
  { perror (modelfile); exit (1); }
  fprintf(modelfl,"SVM-light Version %s\n",VERSION);
  fprintf(modelfl,"%ld # kernel type\n",
	  model->kernel_parm.kernel_type);
  fprintf(modelfl,"%ld # kernel parameter -d \n",
	  model->kernel_parm.poly_degree);
  fprintf(modelfl,"%.8g # kernel parameter -g \n",
	  model->kernel_parm.rbf_gamma);
  fprintf(modelfl,"%.8g # kernel parameter -s \n",
	  model->kernel_parm.coef_lin);
  fprintf(modelfl,"%.8g # kernel parameter -r \n",
	  model->kernel_parm.coef_const);
  fprintf(modelfl,"%s# kernel parameter -u \n",model->kernel_parm.custom);
  fprintf(modelfl,"%ld # highest feature index \n",model->totwords);
  fprintf(modelfl,"%ld # number of training documents \n",model->totdoc);
 
  sv_num=1;
  for(i=1;i<model->sv_num;i++) {
    for(v=model->supvec[i]->fvec;v;v=v->next) 
      sv_num++;
  }
  fprintf(modelfl,"%ld # number of support vectors plus 1 \n",sv_num);
  fprintf(modelfl,"%.8g # threshold b, each following line is a SV (starting with alpha*y)\n",model->b);

  for(i=1;i<model->sv_num;i++) {
    for(v=model->supvec[i]->fvec;v;v=v->next) {
      fprintf(modelfl,"%.32g ",model->alpha[i]*v->factor);
      for (j=0; (v->words[j]).wnum; j++) {
	fprintf(modelfl,"%ld:%.8g ",
		(long)(v->words[j]).wnum,
		(double)(v->words[j]).weight);
      }
      fprintf(modelfl,"#%s\n",v->userdefined);
    /* NOTE: this could be made more efficient by summing the
       alpha's of identical vectors before writing them to the
       file. */
    }
  }
  fclose(modelfl);
  if(verbosity>=1) {
    printf("done\n");
  }
}


MODEL *read_model(char *modelfile)
{
  FILE *modelfl;
  long i,queryid,slackid;
  double costfactor;
  long max_sv,max_words,ll,wpos;
  char *line,*comment;
  WORD *words;
  char version_buffer[100];
  MODEL *model;

  if(verbosity>=1) {
    printf("Reading model..."); fflush(stdout);
  }

  nol_ll(modelfile,&max_sv,&max_words,&ll); /* scan size of model file */
  max_words+=2;
  ll+=2;

  words = (WORD *)my_malloc(sizeof(WORD)*(max_words+10));
  line = (char *)my_malloc(sizeof(char)*ll);
  model = (MODEL *)my_malloc(sizeof(MODEL));

  if ((modelfl = fopen (modelfile, "r")) == NULL)
  { perror (modelfile); exit (1); }

  fscanf(modelfl,"SVM-light Version %s\n",version_buffer);
  if(strcmp(version_buffer,VERSION)) {
    perror ("Version of model-file does not match version of svm_classify!"); 
    exit (1); 
  }
  fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.kernel_type);  
  fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.poly_degree);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.rbf_gamma);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.coef_lin);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.coef_const);
  fscanf(modelfl,"%[^#]%*[^\n]\n", model->kernel_parm.custom);

  fscanf(modelfl,"%ld%*[^\n]\n", &model->totwords);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->totdoc);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->sv_num);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->b);

  model->supvec = (DOC **)my_malloc(sizeof(DOC *)*model->sv_num);
  model->alpha = (double *)my_malloc(sizeof(double)*model->sv_num);
  model->index=NULL;
  model->lin_weights=NULL;

  for(i=1;i<model->sv_num;i++) {
    fgets(line,(int)ll,modelfl);
    if(!parse_document(line,words,&(model->alpha[i]),&queryid,&slackid,
		       &costfactor,&wpos,max_words,&comment)) {
      printf("\nParsing error while reading model file in SV %ld!\n%s",
	     i,line);
      exit(1);
    }
    model->supvec[i] = create_example(-1,
				      0,0,
				      0.0,
				      create_svector(words,comment,1.0));
  }
  fclose(modelfl);
  free(line);
  free(words);
  if(verbosity>=1) {
    fprintf(stdout, "OK. (%d support vectors read)\n",(int)(model->sv_num-1));
  }
  return(model);
}

MODEL *copy_model(MODEL *model)
{
  MODEL *newmodel;
  long  i;

  newmodel=(MODEL *)my_malloc(sizeof(MODEL));
  (*newmodel)=(*model);
  newmodel->supvec = (DOC **)my_malloc(sizeof(DOC *)*model->sv_num);
  newmodel->alpha = (double *)my_malloc(sizeof(double)*model->sv_num);
  newmodel->index = NULL; /* index is not copied */
  newmodel->supvec[0] = NULL;
  newmodel->alpha[0] = 0;
  for(i=1;i<model->sv_num;i++) {
    newmodel->alpha[i]=model->alpha[i];
    newmodel->supvec[i]=create_example(model->supvec[i]->docnum,
				       model->supvec[i]->queryid,0,
				       model->supvec[i]->costfactor,
				       copy_svector(model->supvec[i]->fvec));
  }
  if(model->lin_weights) {
    newmodel->lin_weights = (double *)my_malloc(sizeof(double)*(model->totwords+1));
    for(i=0;i<model->totwords+1;i++) 
      newmodel->lin_weights[i]=model->lin_weights[i];
  }
  return(newmodel);
}

void free_model(MODEL *model, int deep)
{
  long i;

  if(model->supvec) {
    if(deep) {
      for(i=1;i<model->sv_num;i++) {
	free_example(model->supvec[i],1);
      }
    }
    free(model->supvec);
  }
  if(model->alpha) free(model->alpha);
  if(model->index) free(model->index);
  if(model->lin_weights) free(model->lin_weights);
  free(model);
}


void read_documents(char *docfile, DOC ***docs, double **label, 
		    long int *totwords, long int *totdoc)
{
  char *line,*comment;
  WORD *words;
  long dnum=0,wpos,dpos=0,dneg=0,dunlab=0,queryid,slackid,max_docs;
  long max_words_doc, ll;
  double doc_label,costfactor;
  FILE *docfl;

  if(verbosity>=1) {
    printf("Scanning examples..."); fflush(stdout);
  }
  nol_ll(docfile,&max_docs,&max_words_doc,&ll); /* scan size of input file */
  max_words_doc+=2;
  ll+=2;
  max_docs+=2;
  if(verbosity>=1) {
    printf("done\n"); fflush(stdout);
  }

  (*docs) = (DOC **)my_malloc(sizeof(DOC *)*max_docs);    /* feature vectors */
  (*label) = (double *)my_malloc(sizeof(double)*max_docs); /* target values */
  line = (char *)my_malloc(sizeof(char)*ll);

  if ((docfl = fopen (docfile, "r")) == NULL)
  { perror (docfile); exit (1); }

  words = (WORD *)my_malloc(sizeof(WORD)*(max_words_doc+10));
  if(verbosity>=1) {
    printf("Reading examples into memory..."); fflush(stdout);
  }
  dnum=0;
  (*totwords)=0;
  while((!feof(docfl)) && fgets(line,(int)ll,docfl)) {
    if(line[0] == '#') continue;  /* line contains comments */
    if(!parse_document(line,words,&doc_label,&queryid,&slackid,&costfactor,
		       &wpos,max_words_doc,&comment)) {
      printf("\nParsing error in line %ld!\n%s",dnum,line);
      exit(1);
    }
    (*label)[dnum]=doc_label;
    /* printf("docnum=%ld: Class=%f ",dnum,doc_label); */
    if(doc_label > 0) dpos++;
    if (doc_label < 0) dneg++;
    if (doc_label == 0) dunlab++;
    if((wpos>1) && ((words[wpos-2]).wnum>(*totwords))) 
      (*totwords)=(words[wpos-2]).wnum;
    if((*totwords) > MAXFEATNUM) {
      printf("\nMaximum feature number exceeds limit defined in MAXFEATNUM!\n");
      printf("LINE: %s\n",line);
      exit(1);
    }
    (*docs)[dnum] = create_example(dnum,queryid,slackid,costfactor,
				   create_svector(words,comment,1.0));
    /* printf("\nNorm=%f\n",((*docs)[dnum]->fvec)->twonorm_sq);  */
    dnum++;  
    if(verbosity>=1) {
      if((dnum % 100) == 0) {
	printf("%ld..",dnum); fflush(stdout);
      }
    }
  } 

  fclose(docfl);
  free(line);
  free(words);
  if(verbosity>=1) {
    fprintf(stdout, "OK. (%ld examples read)\n", dnum);
  }
  (*totdoc)=dnum;
}

int parse_document(char *line, WORD *words, double *label,
		   long *queryid, long *slackid, double *costfactor,
		   long int *numwords, long int max_words_doc,
		   char **comment)
{
  register long wpos,pos;
  long wnum;
  double weight;
  int numread;
  char featurepair[1000],junk[1000];

  (*queryid)=0;
  (*slackid)=0;
  (*costfactor)=1;

  pos=0;
  (*comment)=NULL;
  while(line[pos] ) {      /* cut off comments */
    if((line[pos] == '#') && (!(*comment))) {
      line[pos]=0;
      (*comment)=&(line[pos+1]);
    }
    if(line[pos] == '\n') { /* strip the CR */
      line[pos]=0;
    }
    pos++;
  }
  if(!(*comment)) (*comment)=&(line[pos]);
  /* printf("Comment: '%s'\n",(*comment)); */

  wpos=0;
  /* check, that line starts with target value or zero, but not with
     feature pair */
  if(sscanf(line,"%s",featurepair) == EOF) return(0);
  pos=0;
  while((featurepair[pos] != ':') && featurepair[pos]) pos++;
  if(featurepair[pos] == ':') {
	perror ("Line must start with label or 0!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
  }
  /* read the target value */
  if(sscanf(line,"%lf",label) == EOF) return(0);
  pos=0;
  while(space_or_null((int)line[pos])) pos++;
  while((!space_or_null((int)line[pos])) && line[pos]) pos++;
  while(((numread=sscanf(line+pos,"%s",featurepair)) != EOF) && 
	(numread > 0) && 
	(wpos<max_words_doc)) {
    /* printf("%s\n",featurepair); */
    while(space_or_null((int)line[pos])) pos++;
    while((!space_or_null((int)line[pos])) && line[pos]) pos++;
    if(sscanf(featurepair,"qid:%ld%s",&wnum,junk)==1) {
      /* it is the query id */
      (*queryid)=(long)wnum;
    }
    else if(sscanf(featurepair,"sid:%ld%s",&wnum,junk)==1) {
      /* it is the slack id */
      if(wnum > 0) 
	(*slackid)=(long)wnum;
      else {
	perror ("Slack-id must be greater or equal to 1!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
      }
    }
    else if(sscanf(featurepair,"cost:%lf%s",&weight,junk)==1) {
      /* it is the example-dependent cost factor */
      (*costfactor)=(double)weight;
    }
    else if(sscanf(featurepair,"%ld:%lf%s",&wnum,&weight,junk)==2) {
      /* it is a regular feature */
      if(wnum<=0) { 
	perror ("Feature numbers must be larger or equal to 1!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
      }
      if((wpos>0) && ((words[wpos-1]).wnum >= wnum)) { 
	perror ("Features must be in increasing order!!!\n"); 
	printf("LINE: %s\n",line);
	exit (1); 
      }
      (words[wpos]).wnum=wnum;
      (words[wpos]).weight=(FVAL)weight; 
      wpos++;
    }
    else {
      perror ("Cannot parse feature/value pair!!!\n"); 
      printf("'%s' in LINE: %s\n",featurepair,line);
      exit (1); 
    }
  }
  (words[wpos]).wnum=0;
  (*numwords)=wpos+1;
  return(1);
}

double *read_alphas(char *alphafile,long totdoc)
     /* reads the alpha vector from a file as written by the
        write_alphas function */
{
  FILE *fl;
  double *alpha;
  long dnum;

  if ((fl = fopen (alphafile, "r")) == NULL)
  { perror (alphafile); exit (1); }

  alpha = (double *)my_malloc(sizeof(double)*totdoc);
  if(verbosity>=1) {
    printf("Reading alphas..."); fflush(stdout);
  }
  dnum=0;
  while((!feof(fl)) && fscanf(fl,"%lf\n",&alpha[dnum]) && (dnum<totdoc)) {
    dnum++;
  }
  if(dnum != totdoc)
  { perror ("\nNot enough values in alpha file!"); exit (1); }
  fclose(fl);

  if(verbosity>=1) {
    printf("done\n"); fflush(stdout);
  }

  return(alpha);
}

void nol_ll(char *file, long int *nol, long int *wol, long int *ll) 
     /* Grep through file and count number of lines, maximum number of
        spaces per line, and longest line. */
{
  FILE *fl;
  int ic;
  char c;
  long current_length,current_wol;

  if ((fl = fopen (file, "r")) == NULL)
  { perror (file); exit (1); }
  current_length=0;
  current_wol=0;
  (*ll)=0;
  (*nol)=1;
  (*wol)=0;
  while((ic=getc(fl)) != EOF) {
    c=(char)ic;
    current_length++;
    if(space_or_null((int)c)) {
      current_wol++;
    }
    if(c == '\n') {
      (*nol)++;
      if(current_length>(*ll)) {
	(*ll)=current_length;
      }
      if(current_wol>(*wol)) {
	(*wol)=current_wol;
      }
      current_length=0;
      current_wol=0;
    }
  }
  fclose(fl);
}

long minl(long int a, long int b)
{
  if(a<b)
    return(a);
  else
    return(b);
}

long maxl(long int a, long int b)
{
  if(a>b)
    return(a);
  else
    return(b);
}

long get_runtime(void)
{
  clock_t start;
  start = clock();
  return((long)((double)start*100.0/(double)CLOCKS_PER_SEC));
}


# ifdef _MSC_VER

int isnan(double a)
{
  return(_isnan(a));
}

# endif 

int space_or_null(int c) {
  if (c==0)
    return 1;
  return isspace((unsigned char)c);
}

void *my_malloc(size_t size)
{
  void *ptr;
  if(size<=0) size=1; /* for AIX compatibility */
  ptr=(void *)malloc(size);
  if(!ptr) { 
    perror ("Out of memory!\n"); 
    exit (1); 
  }
  return(ptr);
}

void copyright_notice(void)
{
  printf("\nCopyright: Thorsten Joachims, thorsten@joachims.org\n\n");
  printf("This software is available for non-commercial use only. It must not\n");
  printf("be modified and distributed without prior permission of the author.\n");
  printf("The author is not responsible for implications from the use of this\n");
  printf("software.\n\n");
}
