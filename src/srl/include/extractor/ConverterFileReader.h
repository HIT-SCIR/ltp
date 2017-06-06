//
// Created by liu on 2017/1/1.
//

#ifndef PROJECT_FILEREADER_H
#define PROJECT_FILEREADER_H

#include "../base/debug.h"
#include <string>
#include "Converter.h"
#include "../structure/DataFileName.h"
#include "../structure/DataFileContext.h"
#include "fstream"
using namespace std;
namespace extractor {
/**
 * 此类是一个接口类，用来定义文件到字符串数组的转换
 * 例1：每行分割无关，行内按照某些字符有特殊含义
 * 例2：每个空行分割无关，行内
 */

  class ConverterFileReader : public Converter<DataFileName, DataFileContext> {
  public:
    base::Debug debug;

    ConverterFileReader() : debug("ConverterFileReader") {};

    virtual void convert(DataFileName & dataFileName) {
      string fileName = dataFileName.fileName;
      debug.debug("reading file '%s'", fileName.c_str());
      ifstream in(fileName);
      if (!in) {
        debug.error("reading file '%s' failed! ", fileName.c_str());
        exit(1);
      }
      string line;
      int lineCounter = 0, blockCounter = 0;
      while (getline(in, line)) {
        if (!(++lineCounter % 10000)) debug.info("%d lines read", lineCounter);
        readLine(line);
      }
      debug.info("read %d lines and get %d blocks", lineCounter, blockCounter);
      debug.debug("finish reading file '%s'", fileName.c_str());
    };

    virtual void readLine(string& line) {
      vector<string> lineFeatures;
      split(lineFeatures, line, boost::is_any_of(" \t"));
      insert(DataFileContext(lineFeatures));
    }


  };
}

#endif //PROJECT_FILEREADER_H
