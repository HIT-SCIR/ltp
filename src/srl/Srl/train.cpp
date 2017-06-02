//
// Created by liu on 2017/4/7.
//

#include "dynet/dynet.h"
#include "base/processLoader.h"
#include "process/TrainSrlSrl.h"

using namespace std;

int main(int argc, char * argv[]) {
  base::ProcessLoader processLoader(argc, argv);
  return processLoader.runProcess<TrainSrlSrl>();
}
