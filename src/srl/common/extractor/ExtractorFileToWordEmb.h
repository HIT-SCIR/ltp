//
// Created by liu on 2017/1/4.
//

#ifndef PROJECT_EXTRACTORFILETOWORDEMB_H
#define PROJECT_EXTRACTORFILETOWORDEMB_H

#include "extractor/AbstractExtractor.h"
#include "structure/DataFileName.h"
#include "ConverterFileContextToWordEmb.h"
#include "extractor/ConverterFileReader.h"
#include "base/debug.h"
using namespace extractor;
using namespace std;

class ExtractorFileToWordEmb: public AbstractExtractor<DataFileName, WordEmb> {
public:
  DataFileName * startPtr;
  base::Debug debug;
  DataFileName file;
  ExtractorFileToWordEmb() : debug("ExtractorFileToWordEmb") { }

  void init(const string& fileName) {
    file = DataFileName(fileName);
    init(file);
  }

  void init(DataFileName &start) {
    startPtr = & start;
  }
  WordEmb run() {
    vector<DataFileName> fileName = {*(startPtr)};
    ConverterFileReader fileReader;
    ConverterFileContextToWordEmb fileContextToWordEmb;
    fileReader.init(fileName);
    fileReader.run();
    fileContextToWordEmb.init(fileReader.getResult());
    fileContextToWordEmb.run();
    WordEmb res = fileContextToWordEmb.getResult();
    return res;
  };
};


#endif //PROJECT_EXTRACTORFILETOWORDEMB_H
