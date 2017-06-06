//
// Created by liu on 2017/1/4.
//

#ifndef PROJECT_CONVERTERFILECONTEXTTOWORDEMB_H
#define PROJECT_CONVERTERFILECONTEXTTOWORDEMB_H

#include "structure/DataFileContext.h"
#include "extractor/AbstractConverter.h"
#include "base/debug.h"
#include "unordered_map"
#include "vector"
#include "base/progressBar.h"
#include "boost/lexical_cast.hpp"

using boost::lexical_cast;
using namespace std;
using namespace extractor;

typedef unordered_map<string, vector<float> > WordEmb;

class ConverterFileContextToWordEmb : public AbstractConverter<vector<DataFileContext>, WordEmb> {
public:
  WordEmb data;
  vector<DataFileContext> * origin;
  base::Debug debug;

  ConverterFileContextToWordEmb(): debug(ConverterFileContextToWordEmb::getClassName()) {}
  virtual void init(vector<DataFileContext> & origin_data) {
    origin = & origin_data;
  }

  virtual void run() {
    int size = origin->size();
    debug.debug("Convert 'DataFileContext' to 'WordEmb' start. total %u lines", size);
    base::ProgressBar bar(size);
    for (int i = 0; i < size; ++i) {
      convert((*origin)[i]);
      if (bar.updateLength(i + 1)) debug.info("(%d)%s lines is converted %d data generate.", i + 1, bar.getProgress(i + 1).c_str() , data.size());
    }
    debug.debug("Convert 'DataFileContext' to 'WordEmb' finish. generate %d WordEmb(s)", data.size());
  }

  virtual void convert(DataFileContext & line) {
    if (line.data.size() < 5)
      return;
    data[line.data[0]] = vector<float>();
    vector<float> & vec = data[line.data[0]];
    for (int j = 1; j < line.data.size(); ++j) {
      if (line.data[j] != "")
        vec.push_back(lexical_cast<float>(line.data[j]));
    }
  };

  virtual WordEmb & getResult() {
    return this->data;
  }

  static string getClassName() {
    return "Converter <DataFileContext, WordEmb>";
  }
};


#endif //PROJECT_CONVERTERFILECONTEXTTOWORDEMB_H
