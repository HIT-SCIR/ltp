//
// Created by liu on 2017/1/2.
//

#ifndef PROJECT_DATAFILEBLOCKCONTEXT_H
#define PROJECT_DATAFILEBLOCKCONTEXT_H

#include "vector"
#include "string"
#include "iostream"
#include "Data.h"
using namespace std;
namespace extractor {
  class DataFileBlockContext : public Data {
  public:
    static string getClassName() { return "DataFileBlockContext"; }
    vector<vector<string> > data;
    /**
     * overwrite
     * @return
     */
    void push_back(vector<string> &block) {
      data.push_back(block);
    }
    void clear() {
      data.clear();
    }
    vector<string> & getLast() {
      return data[data.size() - 1];
    }

  };
}

#endif //PROJECT_DATAFILEBLOCKCONTEXT_H
