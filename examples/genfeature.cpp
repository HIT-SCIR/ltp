//
// Created by jiadongyu on 12/21/15.
//
#include "genfeature.h"

using namespace std;



int main(int argc, char * argv[]){
    omp_set_num_threads(omp_get_num_procs());
    Model model;
    model.main();
}


