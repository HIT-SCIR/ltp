//
// Created by liu on 2017-05-12.
//

#ifndef PROJECT_PIMODEL_H
#define PROJECT_PIMODEL_H

#include <base/debug.h>
#include <model/PiSrlModel.h>
#include "../config/SrlPiConfig.h"
#include "structure/SrlPiSample.h"

// model builders
#include <model/LookupModelBuilder.h>
#include <model/ConstLookupModelBuilder.h>
#include <model/BiRNNModelBuilder.h>
#include <model/AffineTransformModelBuilder.h>
#include "structure/WordEmbBuilder.h"


class PiModel : public PiSrlModel {
  SrlPiBaseConfig & config;
  base::Debug debug;
  // todo define ModelBuilders
  WordEmbBuilder emb_lookup;
  LookupModelBuilder word_lookup, pos_lookup, rel_lookup;
  BiLSTMModelBuilder lstm;
  AffineTransformModelBuilder sentTransform, resultTransform;


public:
  PiModel(SrlPiBaseConfig &config) :
          PiSrlModel(config),
          config(config), debug("PiModel") { }

  void initEmbedding(unordered_map<string, vector<float> > & emb) {
    if (config.emb_dim)
      emb_lookup.setEmb(emb);
  }
  void initEmbedding() {
    if (config.emb_dim)
      emb_lookup.loadEmb(config.embedding);
  }

  void init() {
    vector<unsigned int> sentDims;
    if (config.word_dim) {
      word_lookup = LookupModelBuilder(dict[WORD].size(), config.word_dim); word_lookup.init(model);
      sentDims.push_back(config.word_dim);
    }
    if (config.emb_dim) {
      sentDims.push_back(config.emb_dim);
    }
    if (config.pos_dim) {
      pos_lookup = LookupModelBuilder(dict[POS].size(), config.pos_dim); pos_lookup.init(model);
      sentDims.push_back(config.pos_dim);
    }
    if (config.rel_dim) {
      rel_lookup = LookupModelBuilder(dict[REL].size(), config.rel_dim); rel_lookup.init(model);
      sentDims.push_back(config.rel_dim);
    }
    sentTransform = AffineTransformModelBuilder(sentDims, config.lstm_input_dim); sentTransform.init(model);
    lstm = BiLSTMModelBuilder(config.layers, config.lstm_input_dim, config.lstm_hidden_dim); lstm.init(model);
    resultTransform = AffineTransformModelBuilder({config.lstm_hidden_dim}, 2); resultTransform.init(model);

  }

  virtual vector<Expression> label(ComputationGraph &hg, SrlPiSample &samples) {
    vector<Expression> sents;
    for (int j = 0; j < samples.size(); ++j) {
      vector<Expression> wordFeature;
      if (config.word_dim) {
        wordFeature.push_back(word_lookup.forward(hg, (unsigned) dict[WORD].convert(samples.getWord(j).getWord())));
      }
      if (config.emb_dim) {
        wordFeature.push_back(dynet::expr::input(hg, {config.emb_dim}, emb_lookup.getEmb(samples.getWord(j).getWord())));
      }
      if (config.pos_dim) {
        wordFeature.push_back(pos_lookup.forward(hg, (unsigned) dict[POS].convert(samples.getWord(j).getPos())));
      }
      if (config.rel_dim) {
        wordFeature.push_back(rel_lookup.forward(hg, (unsigned) dict[REL].convert(samples.getWord(j).getRel())));
      }
      sents.push_back(sentTransform.forward(hg, wordFeature));
    }
    lstm.newGraph(hg);
    lstm.startNewSequence();
    vector<Expression> lstm_out = lstm.forward(hg, sents);
    for (int k = 0; k < lstm_out.size(); ++k) {
      lstm_out[k] = softmax(resultTransform.forward(hg, {activate(lstm_out[k])}));
    }
    return lstm_out;
  }

  virtual Expression
  ExtractError(ComputationGraph &hg, vector<Expression> &adists, SrlPiSample &samples, Performance &perf) {
    assert(adists.size() == samples.size());
    vector<Expression> err;
    for (int j = 0; j < adists.size(); ++j) {
      vector<float> ans = as_vector(hg.incremental_forward(adists[j]));
      int is_pred = (int) (samples.getWord(j).getPredicate() == PRED_LABEL);
      setPerf(perf, is_pred, ans);
      err.push_back(pick(log(adists[j]), (unsigned) is_pred));
    }
    return -sum(err);
  }

  virtual void ExtractResults(ComputationGraph &hg, vector<Expression> &adists, SrlPiSample &samples) {
    assert(adists.size() == samples.size());
    for (int j = 0; j < adists.size(); ++j) {
      int god = getMaxId(as_vector(hg.incremental_forward(adists[j])));
      samples.getWord(j).setPredicate((bool) god);
    }
    unsigned long pred_size = samples.getPredicateList().size();
    for (int j = 0; j < adists.size(); ++j) {
      samples.getWord(j).getArgs().resize(pred_size, NIL_LABEL);
    }
  }

};


#endif //PROJECT_PIMODEL_H
