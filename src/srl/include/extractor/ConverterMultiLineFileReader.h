//
// Created by liu on 2017/1/1.
//

#ifndef PROJECT_MULTILINEFILEREADER_H
#define PROJECT_MULTILINEFILEREADER_H


#include "Converter.h"
#include "../structure/DataFileName.h"
#include "../structure/DataFileBlockContext.h"
#include "../base/debug.h"
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "iostream"
#include "string"
#include "fstream"
using namespace std;
using namespace boost;

namespace extractor {
  class ConverterMultiLineFileReader : public Converter<DataFileName, DataFileBlockContext> {
  public:
    base::Debug debug;

    explicit ConverterMultiLineFileReader() : debug("MultiLineFileReader") {};

    virtual void convert(DataFileName & dataFileName) {
      string fileName = dataFileName.fileName;
      debug.debug("reading file '%s'", fileName.c_str());
      ifstream in(fileName);
      string line;
      int lineCounter = 0, blockCounter = 0;
      unsigned blockLength = 0;
      while (getline(in, line)) {
        if (!(++lineCounter % 10000)) debug.info("%d lines read", lineCounter);
        if (line.empty()) {
          if (!(++blockCounter % 1000)) debug.info("%d blocks generate", blockCounter);
          handleMultiLine();
          blockLength = 0;
          continue;
        }
        readLine(line);
        if (blockLength) {
//          if (blockLength != fileBlock.getLast().size()) {
//            debug.error("file format error."); exit(0);
//          }
        } else {
          blockLength = fileBlock.getLast().size();
        }
      }
      if (fileBlock.data.size() != 0) {
        handleMultiLine();
      }
      debug.info("read %d lines and get %d blocks", lineCounter, blockCounter);
      debug.debug("finish reading file '%s'", fileName.c_str());
    };

    virtual void reWriteFile(string fileName) {
      debug.debug("extract data to '%s'", fileName.c_str());
      ofstream out(fileName);
      for (int j = 0; j < data.size(); ++j) {
        for (int k = 0; k < data[j].data.size(); ++k) {
          for (int l = 0; l < data[j].data[k].size(); ++l) {
            out << data[j].data[k][l];
            if (l != data[j].data[k].size() - 1) out << "\t";
          }
          out << endl;
        }
        out << endl;
      }
      out.close();
    }

  protected:
    DataFileBlockContext fileBlock;

    virtual void readLine(string &line) {
      vector<string> lineFeatures;
      split(lineFeatures, line, is_any_of(" \t\n"));
      fileBlock.push_back(lineFeatures);
    }

    virtual void handleMultiLine() {
      insert(fileBlock);
      fileBlock.clear();
    }
  };
}

#endif //PROJECT_MULTILINEFILEREADER_H
