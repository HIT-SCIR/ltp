/************************************************************************/
/*                                                                      */
/*   kernel.h                                                           */
/*                                                                      */
/*   User defined kernel function. Feel free to plug in your own.       */
/*                                                                      */
/*   Copyright: Alessandro Moschitti                                    */
/*   Date: 20.11.06                                                     */
/*                                                                      */
/************************************************************************/

/* KERNEL_PARM is defined in svm_common.h The field 'custom' is reserved for */
/* parameters of the user defined kernel. */
/* Here is an example of custom kernel on a forest and vectors*/                          

// INPUT DESCRIPTION
// The basic input is a set of trees and a set of vectors.
// The semantics of vectors is the following

//        The first vector contains the parameter weights of each tree so its length is num_of_trees.
//        The second vector tells which kind of kernel should be used for trees (i.e. SST or ST) so also its size is num_of_trees.
//        The third vector tells which kind of kernel should be used for feature vectors (i.e. -t from 0 to 3). Its size is num_of_vectors - 4.
//        The fourth vector contains the parameter weights of each vector. Its size is num_of_vectors - 4.
//        From the fith vector to num_of_vectors there are (num_of_vectors - 4) feature vectors that describe the target object.
//
//
//        The final kernel is:   wt[1]*wt'[1]*TK_s1(t1,t'1)+..+wt[n]*wt'[n]*TK_sn(tn,t'n) + 
//                             + wv[1]*wv'[1]*K_r1(v1,v'1)+..+wv[m]*wv'[m]*K_rn(vn,v'n)
//        where:
//               wt[i] and wt'[i] are the weights associated with the i-th trees of the two objects,
//               si is the type of tree kernel applied to i-th trees (i.e. SST with si=1 or ST with si=0),
//               wv[i] and wv'[i] are the weight associated with the i-th feature vectors of the two objects,
//               ri is the type of the kernel applied to the i-th fetature vectors (i.e. ri = 0,1,2,3).
//
//        Example, to evaluate 
//          K(o,o) = 1*ST(t1,t1)+.5*.5*SST(t2,t2)+.1*.1*ST(t3,t3)+.125*.125*poly(v1,v1)+.670*.670*linear(v2,v2),
//        the following data is required (to simplify we have only one object o):
//             +1 |BT|(NN Paul) |BT| (JJ good) |BT| (VB give) |ET| \\ forest
//                    1:1 2:.5 3:.1 |BV| 1:0 2:1 3:0 |BV|          \\ tree parameters
//                    1:.125 2:.670 |BV| 1:1 2:0 |BV|              \\ feature vectors parameters
//                    1132:.2 1300:.01 12234:.23 30000:.23 30001:.001 30023:.034 |BV| \\ feature vectors
//                    4050:.3 5030:.1 11114:.7 |EV|
//
// To test the kernel use the following line as input_file:
// +1 |BT|(NN Paul) |BT| (JJ good) |BT| (VB give) |ET| 1:1 2:.5 3:.1 |BV| 1:0 2:1 3:0 |BV| 1:.125 2:.670 |BV| 1:1 2:0 |BV| 1132:.2 1300:.01 12234:.23 30000:.23 30001:.001 30023:.034 |BV| 4050:.3 5030:.1 11114:.7 |EV|
// and execute the command: svm_learn -t 4  input_file


// implementation 

double custom_kernel(KERNEL_PARM *kernel_parm, DOC *a, DOC *b) 
{

  int i;
  double k;
  
   k=0;

// a and b are structures containing a forest of trees and a set of vectors:
// - forest_vec[i] is the i-th tree
// - vectors[i] is the i-th feature vector
// - num_of_trees
// - num_of_vectors

// summation of tree kernels

   for(i=0; i< a->num_of_trees && i< b->num_of_trees; i++){ // a->num_of_trees should be equal to b->num_of_trees


      if(a->forest_vec[i]!=NULL && b->forest_vec[i]!=NULL){// Test if one the i-th tree of instance a and b is an empty tree

         SIGMA = a->vectors[1]->words[i].weight; // The type of tree kernel for i-th tree is told by vector 1. 
                                                 // The field "weight" according to the input data is 0 (ST) or 1 (SST).
         LAMBDA = 0.4; // An additional vector may contain the lambda parameters instead of .4 for all trees.
                       // other vectors may contain other specific parameters see "struct kernel_parm" in "svm_common.h".
         k+=  // summation of tree kernels
             a->vectors[0]->words[i].weight * // Weight of tree i (vector 0 is used to assign weigths to trees).
             b->vectors[0]->words[i].weight * // Weight of tree i for instace b.
             tree_kernel(kernel_parm, a, b, i, i)/ // Evaluate tree kernel between the two i-th trees.
             sqrt(tree_kernel(kernel_parm, a, a, i, i) * 
                  tree_kernel(kernel_parm, b, b, i, i)); // Normalize respect to both i-th trees.

/* TEST - print the i-th trees (of a and b instances)
printf("\ntree 1: <"); writeTreeString(a->forest_vec[i]->root);  
printf(">\ntree 2: <"); writeTreeString(b->forest_vec[i]->root);printf(">\n"); 
printf("\n\n(i,i)=(%d,%d)= Kernel-Sequence :%f norm1,norm2 (%f,%f)\n",i,i,k,a->forest_vec[i]->twonorm_PT, b->forest_vec[i]->twonorm_PT);
fflush(stdout);
*/
      }

   }
   
// Summation of Vector Kernels

  for(i=0; i< a->num_of_vectors-4 && i< b->num_of_vectors-4; i++)
     
     if(a->vectors[i]!=NULL && b->vectors[i]!=NULL){ // Check if the i-th vectors are empty.
      
        kernel_parm->second_kernel = (long) a->vectors[3]->words[i].weight; // Type of standard feature vector kernel (from 0 to 3).
        kernel_parm->poly_degree = (long) 2; // Set the degree = 2 for polynomial kernel (for linear kernel it does not apply).
                                             // An additional vector could be defined to select different degrees for different feature vectors.
        k=   // summation of vectors
             a->vectors[2]->words[i].weight * // Weight of feature vector i (vector 2 is used to assign weigths to vectors).
             b->vectors[2]->words[i].weight * // Weight of feature vector i for instace b.
             basic_kernel(kernel_parm, a, b, i, i)/ // Compute standard kernel (selected according to the "second_kernel" parameter).
             sqrt(basic_kernel(kernel_parm, a, a, i, i) * 
                  basic_kernel(kernel_parm, b, b, i, i)); //normalize vectors

//TEST printf("\n\n(i,i)=(%d,%d)= Kernel-Sequence :%f norm1,norm2 (%f,%f)\n",i,i,k,a->vectors[i]->twonorm_STD, b->vectors[i]->twonorm_STD);

      }
 
   return k;
}
