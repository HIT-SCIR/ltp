//
// Created by liu on 2016/12/19.
//

#include "debug.h"

base::Debug::Debug(string modelName):modelName(modelName) {
  if (enabledModels.count(modelName) || enabledModels.count("*")) {
    disable = false;
  }
}

void base::Debug::init(base::DebugConfig config)  {
  logLevel = config.logLevel;
  vector<string> enModelsVec;
  boost::split(enModelsVec, config.enabledModels, boost::is_any_of(","));
  enabledModels.clear();
  for (auto i :enModelsVec) {
    enabledModels.insert(i);
  }
}

int base::Debug::logLevel = (int)LogLevel::info;
char* base::Debug::tmpBuffer = new char[4096];
set<string> base::Debug::enabledModels;

