//
// Created by liu on 2017-05-12.
//

#ifndef Srl_Pi_CONFIG_H
#define Srl_Pi_CONFIG_H

#include "config/ModelConf.h"
#include "boost/serialization/access.hpp"

class SrlPiBaseConfig : public virtual ModelConf {
public:
  unsigned word_dim;
  unsigned emb_dim;
  unsigned pos_dim;
  unsigned rel_dim;
  unsigned lstm_input_dim;
  unsigned lstm_hidden_dim;
  unsigned layers;

  string embedding;

  SrlPiBaseConfig(string confName = "Configuration"): ModelConf(confName) {
    registerConf<unsigned>("word_dim"       , UNSIGNED, word_dim         , "word dimension"       , 100);
    registerConf<unsigned>("emb_dim"        , UNSIGNED, emb_dim          , "embedding dimension"  , 50);
    registerConf<unsigned>("pos_dim"        , UNSIGNED, pos_dim          , "postag dimension"     , 12);
    registerConf<unsigned>("rel_dim"        , UNSIGNED, rel_dim          , "relation dim"         , 50);
    registerConf<unsigned>("lstm_input_dim" , UNSIGNED, lstm_input_dim   , "lstm_input_dim"       , 100);
    registerConf<unsigned>("lstm_hidden_dim", UNSIGNED, lstm_hidden_dim  , "lstm_hidden_dim"      , 100);
    registerConf<unsigned>("layers"         , UNSIGNED, layers           , "lstm layers"          , 1);

    registerConf<string>  ("embedding" , STRING,   embedding , "embedding", "");
  }

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int) {
    ar & word_dim;
    ar & emb_dim;
    ar & pos_dim;
    ar & rel_dim;
    ar & lstm_input_dim;
    ar & lstm_hidden_dim;
    ar & layers;
  }
};

class SrlPiTrainConfig : public virtual SrlPiBaseConfig, public virtual LabelModelTrainerConf {
public:

  SrlPiTrainConfig(string confName = "Configuration"):
          SrlPiBaseConfig(confName),
          LabelModelTrainerConf(confName)
  {

  }
};

class SrlPiPredConfig : public virtual SrlPiBaseConfig, public virtual LabelModelPredictorConf {
public:
  SrlPiPredConfig(string confName = "Configuration"):
          SrlPiBaseConfig(confName),
          LabelModelPredictorConf(confName)
  { }
};

#endif //Srl_Pi_CONFIG_H
