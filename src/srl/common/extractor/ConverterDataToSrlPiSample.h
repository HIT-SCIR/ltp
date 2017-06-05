//
// Created by liu on 2017-05-12.
//

#ifndef Srl_CONVENTERDATATO_Pi_SAMPLE_H
#define Srl_CONVENTERDATATO_Pi_SAMPLE_H

#include "extractor/BiConverter.h"
#include "structure/DataFileBlockContext.h"
#include "../structure/SrlPiSample.h"
using namespace extractor;

class ConverterDataToSrlPiSample : public BiConverter<DataFileBlockContext, SrlPiSample>{
public:
  ConverterDataToSrlPiSample() {}

  virtual void convert(DataFileBlockContext &t1) {
    SrlPiSample sample;
    for (int j = 0; j < t1.data.size(); ++j) {
      sample.push_back(convertLineToWord(t1.data[j]));
    }
    if (sample.getPredicateList().size()) {
      data.push_back(sample);
    }
  }

  inline Word convertLineToWord(vector<string> & line) {
    // 0  1   2   3   4 5 6 7 8 9 0   11  1213      14
    // 4	有人	有人	有人	r	r	_	_	5	5	SBV	SBV	Y	有人.01	A1	_	_	_	_	_	_	_
    int index = lexical_cast<int>(line[0]) - 1;
    int parent = lexical_cast<int>(line[8]) - 1;
    vector<string> labels;
    // 14 位置开始论元标号
    for (int j = 14 ; j < line.size(); ++j) {
      labels.push_back(line[j]);
    }
    Word word(index, line[1], line[4], parent, line[10] ,(index <= parent ? "before" : "after"), line[12], labels);
    return word;
  }

  virtual void iconvOne(DataFileBlockContext &t1, SrlPiSample &t2, int innerIndex) {
    assert(t1.data.size() == t2.size());
    for (int j = 0; j < t2.size(); ++j) {
      vector<string>& line = t1.data[j];
      line[12] = t2.getWord(j).getPredicate();
      line[13] = (line[12] == PRED_LABEL ? line[1] + ".01": NIL_LABEL);
      line.erase(line.begin() + 14, line.end());
      line.insert(line.end(), t2.getWord(j).getArgs().begin(), t2.getWord(j).getArgs().end());
    }
  }

};

#endif //Srl_CONVENTERDATATO_Pi_SAMPLE_H
