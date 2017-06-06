//
// Created by liu on 2017-05-12.
//

#ifndef Srl_Srl_CONFIG_H
#define Srl_Srl_CONFIG_H

#include "config/ModelConf.h"

class SrlSrlBaseConfig : public virtual ModelConf {
public:
  unsigned word_dim;
  unsigned emb_dim;
  unsigned pos_dim;
  unsigned rel_dim;
  unsigned position_dim;
  unsigned lstm_input_dim;
  unsigned lstm_hidden_dim;
  unsigned hidden_dim;
  unsigned layers;
  string embedding;
  SrlSrlBaseConfig(string confName = "Configuration"): ModelConf(confName) {
    registerConf<unsigned>("word_dim"       , UNSIGNED, word_dim         , "word dimension"       , 100);
    registerConf<unsigned>("emb_dim"        , UNSIGNED, emb_dim          , "embedding dimension"  , 50);
    registerConf<unsigned>("pos_dim"        , UNSIGNED, pos_dim          , "postag dimension"     , 12);
    registerConf<unsigned>("rel_dim"        , UNSIGNED, rel_dim          , "relation dimension"   , 50);
    registerConf<unsigned>("position_dim"   , UNSIGNED, position_dim     , "position dimension"   , 5);
    registerConf<unsigned>("lstm_input_dim" , UNSIGNED, lstm_input_dim   , "lstm_input_dim"       , 100);
    registerConf<unsigned>("lstm_hidden_dim", UNSIGNED, lstm_hidden_dim  , "lstm_hidden_dim"      , 100);
    registerConf<unsigned>("hidden_dim"     , UNSIGNED, hidden_dim       , "Hidden state dimension",100);
    registerConf<unsigned>("layers"         , UNSIGNED, layers           , "lstm layers"          , 1);

    registerConf<string>  ("embedding" , STRING,   embedding , "word embedding file", "");
  }


  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int) {
    ar & word_dim;
    ar & emb_dim;
    ar & pos_dim;
    ar & rel_dim;
    ar & position_dim;
    ar & lstm_input_dim;
    ar & lstm_hidden_dim;
    ar & hidden_dim;
    ar & layers;
  }
};

class SrlSrlTrainConfig : public virtual SrlSrlBaseConfig, public virtual LabelModelTrainerConf {
public:

  SrlSrlTrainConfig(string confName = "Configuration"):
          SrlSrlBaseConfig(confName),
          LabelModelTrainerConf(confName)
  { }
};

class SrlSrlPredConfig : public virtual SrlSrlBaseConfig, public virtual LabelModelPredictorConf {
public:
  SrlSrlPredConfig(string confName = "Configuration"):
          SrlSrlBaseConfig(confName),
          LabelModelPredictorConf(confName)
  { }
};

#endif //Srl_Srl_CONFIG_H
