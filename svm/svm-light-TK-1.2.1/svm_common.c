/************************************************************************/
/*                                                                      */
/*   svm_common.c                                                       */
/*                                                                      */
/*   Definitions and functions used in both svm_learn and svm_classify. */
/*                                                                      */
/*   Author: Thorsten Joachims                                          */
/*   Date: 02.07.02                                                     */
/*                                                                      */
/*                                                                      */
/*                                                                      */
/*   Modified by Alessandro Moschitti for adding the Tree Kernels	*/
/*   Date: 21.10.04 							*/
/*                                                                      */
/*                                                                      */
/*   Copyright (c) 2002  Thorsten Joachims - All rights reserved        */
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

  dist=0;
  for(i=1;i<model->sv_num;i++) {  
    dist+=kernel(&model->kernel_parm,model->supvec[i],ex)*model->alpha[i];
  }
  return(dist-model->b);
}

double classify_example_linear(MODEL *model, DOC *ex) 
     /* classifies example for linear kernel */
     
     /* important: the model must have the linear weight vector computed */

     /* important: the feature numbers in the example to classify must */
     /*            not be larger than the weight vector!               */
{
  return((double)(sprod_ns(model->lin_weights,ex->vectors[0]->words)-model->b));
}

CFLOAT kernel(KERNEL_PARM *kernel_parm, DOC *a, DOC *b) 
     /* calculate the kernel function */
{
    // double k1;
  kernel_cache_statistic++;
  switch(kernel_parm->kernel_type) {
    case 0: /* linear */ 
            return(CFLOAT)sprod(a, b); 
    case 1: /* polynomial */   
            return ((CFLOAT)pow(kernel_parm->coef_lin*sprod(a, b)
                                +kernel_parm->coef_const,(double)kernel_parm->poly_degree));
    case 2: /* radial basis function */
            return((CFLOAT)exp(-kernel_parm->rbf_gamma*(a->twonorm_sq-2*sprod(a,b)+b->twonorm_sq)));
    case 3: /* sigmoid neural net */
            return((CFLOAT)tanh(kernel_parm->coef_lin*sprod(a,b)+kernel_parm->coef_const)); 
    case 4: /* custom-kernel supplied in file kernel.h */
            return((CFLOAT)custom_kernel(kernel_parm,a,b));
    case 5: /* combine kernels */
            return ((CFLOAT)advanced_kernels(kernel_parm,a,b));
    case 11: /* Re-ranking of predicate argument structures only trees, i.e. re-ranking using trees */
            return((CFLOAT)SRL_re_ranking_CoNLL2006(kernel_parm,a,b));           
    case 12: /* Re-ranking of predicate argument structures only trees + vectors */
            return((CFLOAT)SRL_re_ranking_CoNLL2006(kernel_parm,a,b));            
    case 13: /* Re-ranking of predicate argument structures only vectors */
            return((CFLOAT)SRL_re_ranking_CoNLL2006(kernel_parm,a,b)); 
    case 14: /* Classification of Entailment pairs */
            return((CFLOAT)ACL2006_Entailment_kernel(kernel_parm,a,b)); 
          

default: /* Advanced vectorial kernels*/
             printf("\nNo kernel corresponding to -t = %ld option \n",kernel_parm->kernel_type);
             exit(-1);
 
/*if (b->num_of_trees > 0 && a->num_of_trees>0) {
      printf("\n tree 1: <"); writeTreeString(a->forest_vec[0]->root);
                       printf(">\n tree 2: <"); writeTreeString(b->forest_vec[0]->root);printf(">\n");
                      printf("Kernel :%1.20lf \n",k1);  
 }*/
      }
}

double sprod(DOC *a, DOC *b){ // compatibility with standard svm-light
       if(a->num_of_vectors>0 && b->num_of_vectors>0 ){
             if(a->vectors[0]==NULL || b->vectors[0]==NULL){
                printf("ERROR: first vector not defined (with a traditional kernel it must be defined)\n");
                exit(-1);
             }
             else return sprod_ss(a->vectors[0]->words,b->vectors[0]->words);
       }      
       return 0;
}

double sprod_i(DOC *a, DOC *b, int i, int j){ // compatibility with standard svm-light
       if(a->num_of_vectors>0 && b->num_of_vectors>0 ){
             if(a->vectors[i]==NULL || b->vectors[j]==NULL){
                printf("ERROR: first vector not defined (with a traditional kernel it must be defined)\n");
                exit(-1);
             }
             else return sprod_ss(a->vectors[i]->words,b->vectors[j]->words);
       }      
       return 0;
}

double sprod_ss(WORD *a, WORD *b)
     /* compute the inner product of two sparse vectors */
{
    register FVAL sum=0;
    register WORD *ai,*bj;
    ai=a;
    bj=b;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	ai++;
      }
      else {
	sum+=ai->weight * bj->weight;
	ai++;
	bj++;
      }
    }
    return((double)sum);

return 1;
}

WORD* sub_ss(WORD *a, WORD *b) 
     /* compute the difference a-b of two sparse vectors */
{
    register WORD *sum,*sumi;
    register WORD *ai,*bj;
    long veclength;
  
    ai=a;
    bj=b;
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
    ai=a;
    bj=b;
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
    return(sum);
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

void add_vector_ns(double *vec_n, WORD *vec_s, double faktor)
{
  register WORD *ai;
  ai=vec_s;
  while (ai->wnum) {
    vec_n[ai->wnum]+=(faktor*ai->weight);
    ai++;
  }
}

double sprod_ns(double *vec_n, WORD *vec_s)
{
  register double sum=0;
  register WORD *ai;
  ai=vec_s;
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

  model->lin_weights=(double *)my_malloc(sizeof(double)*(model->totwords+1));
  clear_vector_n(model->lin_weights,model->totwords);
  for(i=1;i<model->sv_num;i++) {
    add_vector_ns(model->lin_weights,(model->supvec[i]->vectors[0])->words,
		  model->alpha[i]);
  }
}


void read_model(char *modelfile, MODEL *model, long  max_words_doc, long int ll)
{
  FILE *modelfl;
  int i;
  int pos;
  char *line;
  char version_buffer[100];
  long int fake_totwords;
    
  if(verbosity>=1) {
    printf("Reading model..."); fflush(stdout);
  }

  line = (char *)my_malloc(sizeof(char)*ll);
 
  if ((modelfl = fopen (modelfile, "r")) == NULL)
  { perror (modelfile); exit (1); }

  fscanf(modelfl,"SVM-light Version %s\n",version_buffer);

//printf("Version file %s --- version label %s \n",version_buffer,VERSION);


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
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.lambda);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.tree_constant);
  fscanf(modelfl,"%c%*[^\n]\n", &model->kernel_parm.combination_type);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.first_kernel);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.second_kernel);
  fscanf(modelfl,"%f%*[^\n]\n", &model->kernel_parm.sigma);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.normalization);

  fscanf(modelfl,"%c%*[^\n]\n", &model->kernel_parm.vectorial_approach_standard_kernel);
  fscanf(modelfl,"%c%*[^\n]\n", &model->kernel_parm.vectorial_approach_tree_kernel);

  fscanf(modelfl,"%ld%*[^\n]\n", &model->totwords);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->totdoc);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->sv_num);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->b);


  LAMBDA = model->kernel_parm.lambda; // to make faster the kernel evaluation 
  SIGMA = model->kernel_parm.sigma;   // use global variables

  for(i=1;i<model->sv_num;i++) {
    fgets(line,(int)ll,modelfl);
    pos=0;
    while(strlen(line)<3){
	printf("\nWARNING: empty line for the support vector %d\n\n",i);    
	fgets(line,(int)ll,modelfl);
    }
    sscanf(line,"%lf",&model->alpha[i]);
    model->supvec[i] = (DOC *)my_malloc(sizeof(DOC));

 //   printf("--->%d--><%s>\n",i,line+pos);fflush(stdout);
 //   printf("Go after alpha\n");fflush(stdout);
    
    while(!isspace((int)line[pos]) && line[pos]!=0)pos++; //go after alpha

 //   printf("--->%d--><%s>\n",i,line+pos);fflush(stdout);
 //   printf("Go after spaces\n");fflush(stdout);
  
    while(isspace((int)line[pos])) pos++;// go after spaces
 //   printf("--->%d--><%s>\n",i,line+pos);fflush(stdout);

    read_tree_forest(line, model->supvec[i], &pos);
  
/* Look for Standard Features */
// writeTreeString((model->supvec[i])->root);

  //printf("LINE:%s\n",line+pos);fflush(stdout);
  while(isspace((int)line[pos])&& line[pos]!=0) pos++;//remove spaces
  //while((!isspace((int)line[pos])) && line[pos]) pos++;
  // if vectors exists
  if(line[pos]!=0) read_vector_set(line+pos, model->supvec[i],max_words_doc,&fake_totwords);// read the set of vectors 
  else {model->supvec[i]->vectors=NULL;model->supvec[i]->num_of_vectors=0;}

  evaluateNorma(&(model->kernel_parm), model->supvec[i]);
  (model->supvec[i])->docnum = -1;
  }
  fclose(modelfl);
  free(line);
   if(verbosity>=1) {
    fprintf(stdout, "OK. (%d support vectors read)\n",(int)(model->sv_num-1));
  }
}

void read_documents(char *docfile, DOC *docs, double *label, 
		    long int max_words_doc, long int ll, 
		    long int *totwords, long int *totdoc, KERNEL_PARM *kernel_parm)
{
  char *line;
  DOC doc;
  long dnum=0,dpos=0,dneg=0,dunlab=0;
  double doc_label;
  FILE *docfl;

  line = (char *)my_malloc(sizeof(char)*ll);

  if ((docfl = fopen (docfile, "r")) == NULL)
  { perror (docfile); exit (1); }

  if(verbosity>=1) {
    printf("Reading examples into memory..."); fflush(stdout);
  }
  dnum=0;
  (*totwords)=0;
  
  while((!feof(docfl)) && fgets(line,(int)ll,docfl)) {

      if(strlen(line)==0){
         printf("\nERROR: empty line, missing end of line before end of file\n");
         exit(1);
      }
         

      if(!parse_document(line, &doc, &doc_label, totwords, max_words_doc, kernel_parm)) {
         printf("\nParsing error in line %ld!\n%s",dnum,line);
         exit(1);
      }

    label[dnum]=doc_label;
    /*  printf("Class=%ld ",doc_label);  */
    if (doc_label > 0) dpos++;
    if (doc_label < 0) dneg++;
    if (doc_label == 0) dunlab++;

    docs[dnum].queryid = doc.queryid;
    docs[dnum].costfactor = doc.costfactor;
    
    docs[dnum].forest_vec = doc.forest_vec;
	docs[dnum].num_of_trees = doc.num_of_trees;
	docs[dnum].vectors = doc.vectors;
	docs[dnum].num_of_vectors = doc.num_of_vectors;
      
    // less than 5 basic kernels and greater than 50 only vectors (to save memory)
    if (kernel_parm->kernel_type<4) { // from 0 to 3 are original kernels => no trees
       freeForest(&doc); // save memory by freeing trees
       docs[dnum].num_of_trees = 0;
       docs[dnum].forest_vec =NULL;
       kernel_parm->second_kernel=kernel_parm->kernel_type;
    }   
 
    // establish some interval to free vectors
    
//    if(kernel_parm->kernel_type>20){
//	     docs[dnum].vectors = NULL;
//         docs[dnum].num_of_vectors = 0;
//         freeVectorSet(&doc); // save memory by freeing vectors
//     }
      
    docs[dnum].docnum=dnum;

/* printf("\nNorm=%f\n",docs[dnum].twonorm_sq);  */

/*printf("parse tree number %d: ",dnum);
    writeTreeString(doc.root);
*/
/*    printf("%d\t",(int)doc_label);  
*/ 
    dnum++;  
    if(verbosity>=1) {
      if((dnum % 100) == 0) {
	           printf("%ld..",dnum); fflush(stdout);
      }
    }
  } 

  fclose(docfl);
  free(line);

  if(verbosity>=1) {
    fprintf(stdout, "OK. (%ld examples read)\n", dnum);
  }

fflush(stdout);

  (*totdoc)=dnum;
}

int parse_document(char *line, DOC *doc, double *label, 
		   long int *totwords, long int max_words_doc, KERNEL_PARM *kernel_parm)
{
  int pos;
  
  doc->queryid=0;
  doc->costfactor=1;
 //printf("\n\n---------------------------------------------------------\n\n");fflush(stdout);
  pos=0;

 // while((isspace((int)line[pos])||line[pos]=='\n') && line[pos]!=0) 
   //    {printf("%c\t",line[pos]);pos++;}
  
  if(sscanf(line+pos,"%lf",label) == EOF) return(0);
  // printf("LINE:%s\n\n\n",line);fflush(stdout);

  while(!isspace((int)line[pos]))pos++;// go after label 
  
  read_tree_forest(line, doc, &pos);// read the tree forest

  /* Look for Standard Features, pos returns the end of the last tree*/
//  printf("LINE:%s\n",line+pos);fflush(stdout);

  while(isspace((int)line[pos])&& line[pos]!=0) pos++;// go to "|BV|" marker or to the first number
  /*while((!isspace((int)line[pos])) && line[pos]) pos++;*/
   if(line[pos]!=0) read_vector_set(line+pos, doc, max_words_doc, totwords);// read the tree forest 
     else {doc->vectors=NULL;doc->num_of_vectors=0;}

  doc->docnum=-1;
 
  evaluateNorma(kernel_parm,doc);
  return(1);
}


/*void go_after_STD_mark(FILE *f1, long int *ll){
  char mark[1000];
  *mark=0;
  while(strstr(mark,"|ET|")!=NULL && !feof(f1)){
	//printf("%s\n",mark);
	fscanf(f1,"%s",mark);
	(*ll)+=strlen(mark)+4; // count some spaces more
	*mark=0;
  }
 // ...no problem if the program will double the memory it is just for a line;
}
*/

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
 // go_after_STD_mark(fl,ll);
  while((ic=getc(fl)) != EOF) {
    c=(char)ic;
    current_length++;
    if(isspace((int)c)) {
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
	// printf ("%d %d\n",current_wol,current_length);
      current_length=0;
      current_wol=0;
    }
  }
  fclose(fl);

  if(current_length>(*ll)) {
	(*ll)=current_length;
      }
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


# ifdef MICROSOFT

int isnan(double a)
{
  return(_isnan(a));
}

# endif


void *my_malloc(size_t size)
{
  void *ptr;
  ptr=(void *)malloc(size);
  if(!ptr) { 
    perror ("Out of memory!\n"); 
    exit (1); 
  }
  return(ptr);
}

void copyright_notice(void)
{
  printf("\nCopyright: Thorsten Joachims, thorsten@ls8.cs.uni-dortmund.de\n\n");
  printf("This software is available for non-commercial use only. It must not\n");
  printf("be modified and distributed without prior permission of the author.\n");
  printf("The author is not responsible for implications from the use of this\n");
  printf("software.\n\n");
}

/* DEBUG

            double k1;  WORD *pippo;   

printf("doc IDs :%d %d  ",a->docnum,b->docnum);
if(a->vectors!=NULL && b->vectors!=NULL){
pippo=a->vectors[0]->words;
while(pippo->wnum!=0){printf ("%ld:%lf ",pippo->wnum,pippo->weight);pippo++;};fflush(stdout);
pippo=b->vectors[0]->words;printf("\t");
while(pippo->wnum!=0){printf ("%ld:%lf ",pippo->wnum,pippo->weight);pippo++;};fflush(stdout);
printf("  KERNEL %lf\n",k1);
}

            return k1; 
*/
