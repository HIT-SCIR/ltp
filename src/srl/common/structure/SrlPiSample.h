//
// Created by liu on 2017-05-12.
//

#ifndef PROJECT_SrlPiSAMPLE_H
#define PROJECT_SrlPiSAMPLE_H

#include "vector"
#include "Word.h"
#include "structure/DataConcept.h"
using namespace std;

class SrlPiSample : public extractor::DataConcept {
  vector<Word> data;
  static Word root;
public:
  unsigned size() {
    return data.size();
  }

  vector<int> getPredicateList() {
    vector<int> ans;
    for (int j = 0; j < data.size(); ++j) {
      if (data[j].isPredicate()) {
        ans.push_back(data[j].getInnerIndex());
      }
    }
    return ans;
  }

  Word & getWord(int index) {
    if (index == -1) return root;
    return data[index];
  }

  void push_back(const Word & w) {
    data.push_back(w);
  }

  static string getClassName() {
    return "PiSample";
  }
};


#endif //PROJECT_PiSAMPLE_H
