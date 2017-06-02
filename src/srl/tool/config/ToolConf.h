//
// Created by liu on 2017/2/22.
//

#ifndef PROJECT_TOOLCONF_H
#define PROJECT_TOOLCONF_H

#include "base/config.h"
#include "base/debug.h"

class MergerConfig : virtual public base::DebugConfig {
public:
  string pi_config;
  string srl_config;
  string pi_model;
  string srl_model;
  string embedding;
  string out_model;
  MergerConfig(string confName = "Configuration"): base::DebugConfig(confName) {
    registerConf<string> ("pi_config", STRING, pi_config, "pi_config");
    registerConf<string> ("srl_config", STRING, srl_config, "srl_config");
    registerConf<string> ("pi_model", STRING, pi_model, "pi_model");
    registerConf<string> ("srl_model", STRING, srl_model, "srl_model");
    registerConf<string> ("embedding", STRING, embedding, "embedding");
    registerConf<string> ("out_model", STRING, out_model, "out_model");
  }

};

#endif //PROJECT_TOOLCONF_H
