//
// Created by liu on 2017-05-12.
//

#ifndef PROJECT_PREDSTNLSTM_H
#define PROJECT_PREDSTNLSTM_H

#include <process/LabelModelSGDSeqPredictor.h>
#include <extractor/ConverterMultiLineFileReader.h>
#include "../config/SrlSrlConfig.h"
#include "structure/SrlPiSample.h"
#include "../model/SrlSrlModel.h"
#include "extractor/ConverterDataToSrlPiSample.h"

using namespace std;

class PredSrlSrl : public LabelModelSGDSeqPredictor<SrlSrlPredConfig, SrlPiSample>{
  SrlSrlPredConfig & config;
  SrlSrlModel model;
  ConverterDataToSrlPiSample conv_toSample;
  extractor::ConverterMultiLineFileReader fileReader;
public:
  PredSrlSrl(SrlSrlPredConfig &config)
          : LabelModelSGDSeqPredictor(config, model),
            config(config), model(config) {}

  virtual void init() {

    // todo prepair
    // todo 1. prepair testSamples (to fill 'testSamples')
    initSample(testSamples, config.test_data);
    model.loadDict(); // load dict
    model.init(); // init parameters
    model.load(); // load model
    model.initEmbedding();
  }

  virtual void extractResult() {
    conv_toSample.iconv(&testSamples);
    fileReader.reWriteFile(config.output);
  }

private:

  void initSample(vector<SrlPiSample>& samples, string file) {
    vector<extractor::DataFileName> fileName = {file};
    fileReader.init(fileName);
    fileReader.run();

    conv_toSample.init(fileReader.getResult());
    conv_toSample.run();
    samples = conv_toSample.getResult();
  }
};


#endif //PROJECT_PREDSTNLSTM_H
