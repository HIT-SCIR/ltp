//
// Created by liu on 2017/5/24.
//

#include "process/merge.h"
#include "base/processLoader.h"

using namespace std;
int main(int argc, char * argv[]) {
  base::ProcessLoader processLoader(argc, argv);
  return processLoader.runProcess<Merge>();
}