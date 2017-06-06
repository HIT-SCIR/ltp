//
// Created by liu on 2017/5/24.
//

#ifndef BILSTM_SRL_MERGE_H
#define BILSTM_SRL_MERGE_H

#include <extractor/ExtractorFileToWordEmb.h>
#include "base/process.h"
#include "../config/ToolConf.h"
#include "dynet/dynet.h"
#include "../../Srl/config/SrlSrlConfig.h"
#include "../../Pi/config/SrlPiConfig.h"
#include "../../Pi/model/SrlPiModel.h"
#include "../../Srl/model/SrlSrlModel.h"

class Merge : public base::Process <MergerConfig>{
public:
  Merge(MergerConfig &config) : base::Process <MergerConfig>(config){}

  virtual void main() {
      dynet::DynetParams params;
      params.mem_descriptor = "2000";
      dynet::initialize(params);

      SrlPiBaseConfig  piConfig;
      SrlSrlBaseConfig srlConfig;
      piConfig.init(config.pi_config); piConfig.model = config.pi_model;
      srlConfig.init(config.srl_config); srlConfig.model = config.srl_model;

      ExtractorFileToWordEmb conv;
      conv.init(config.embedding);
      unordered_map<string, vector<float>> embedding = conv.run();

      PiModel pi_model(piConfig);
      pi_model.loadDict();
      pi_model.init();
      pi_model.load();
      pi_model.initEmbedding(embedding);

      SrlSrlModel srl_model(srlConfig);
      srl_model.loadDict();
      srl_model.init();
      srl_model.load();
      srl_model.initEmbedding(embedding);

      ofstream out(config.out_model);
      boost::archive::binary_oarchive oa(out);
      oa << piConfig;
      oa << srlConfig;
      oa << embedding;
      pi_model.save(oa);
      srl_model.save(oa);
      out.close();
  }
};


#endif //BILSTM_SRL_MERGE_H
