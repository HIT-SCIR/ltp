//
// Created by liu on 2017/1/2.
//

#ifndef PROJECT_DATAFILENAME_H
#define PROJECT_DATAFILENAME_H

#include "string"
#include "iostream"
#include "Data.h"
using namespace std;
namespace extractor {
  class DataFileName: public Data {
  public:
    static string getClassName() { return "DataFileName"; }
    string fileName;
    DataFileName() {};

    DataFileName(const string &fileNameString) {
      fileName = fileNameString;
    };
  };
}
#endif //PROJECT_DATAFILENAME_H
