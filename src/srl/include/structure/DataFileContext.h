//
// Created by liu on 2017/1/2.
//

#ifndef PROJECT_DATAFILECONTEXT_H
#define PROJECT_DATAFILECONTEXT_H

#include "vector"
#include "string"
#include "iostream"
#include "Data.h"
using namespace std;
namespace extractor {
  class DataFileContext : public Data {
  public:
    static string getClassName() { return "DataFileContext"; }
    vector<string> data;
    DataFileContext() {};
    DataFileContext(vector<string>& data): data(data) {}
  };
}

#endif //PROJECT_DATAFILECONTEXT_H
