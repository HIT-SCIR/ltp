//
// Created by liu on 2017-05-12.
//

#ifndef PROJECT_STNLSTM_H
#define PROJECT_STNLSTM_H

#include <vector>
#include <base/debug.h>
#include <model/SeqLabelModel.h>
#include "../config/SrlSrlConfig.h"
#include "structure/SrlPiSample.h"


// model builders
#include <model/LookupModelBuilder.h>
#include <model/ConstLookupModelBuilder.h>
#include <model/BiRNNModelBuilder.h>
#include <model/AffineTransformModelBuilder.h>
#include <model/PiSrlModel.h>
#include <structure/WordEmbBuilder.h>


class SrlSrlModel : public PiSrlModel {
  SrlSrlBaseConfig & config;
  base::Debug debug;
  // todo define ModelBuilders
  WordEmbBuilder emb_lookup;
  LookupModelBuilder word_lookup, pos_lookup, rel_lookup, position_lookup;
  BiLSTMModelBuilder ctx_sent_lstm, stx_sent_lstm, stx_rel_lstm;
  AffineTransformModelBuilder sentTransform, hiddenTransform, resultTransform;


public:
  SrlSrlModel(SrlSrlBaseConfig &config) :
          PiSrlModel(config), config(config), debug("SrlSrlModel") { }

  void initEmbedding(unordered_map<string, vector<float> > & emb) {
    if (config.emb_dim)
      emb_lookup.setEmb(emb);
  }
  void initEmbedding() {
    if (config.emb_dim)
      emb_lookup.loadEmb(config.embedding);
  }

  void init();

  virtual vector<Expression> label(ComputationGraph &hg, SrlPiSample &samples);

  vector<Expression> labelOnePredicate(ComputationGraph &hg, SrlPiSample &samples, int predIndex);

  void getStnPath(SrlPiSample &samples, int predIndex, int argIndex, vector<int>& predPath, vector<int>& argPath);

  virtual Expression
  ExtractError(ComputationGraph &hg, vector<Expression> &adists, SrlPiSample &samples, Performance &perf) {
    // todo define your loss
    vector<Expression> err;
    vector<int> predicates = samples.getPredicateList();
    assert(samples.size() * predicates.size() == adists.size());
    int w_size = samples.size(), p_size = (int) predicates.size();
    for (int pi = 0; pi < p_size; pi++) {
      for (int wi = 0; wi < w_size; ++wi) {
        int god = dict[ARG].convert(samples.getWord(wi).getArgs()[pi]);
        setPerf(perf, god, as_vector(hg.incremental_forward(adists[pi * w_size + wi])), dict[ARG].convert(NIL_LABEL));
        err.push_back(log(pick(adists[pi * w_size + wi], god)));
      }
    }
    return -sum(err);
  }

  virtual void ExtractResults(ComputationGraph &hg, vector<Expression> &adists, SrlPiSample &samples) {
    vector<int> predicates = samples.getPredicateList();
    assert(samples.size() * predicates.size() == adists.size());
    int w_size = samples.size(), p_size = (int) predicates.size();
    for (int wi = 0; wi < w_size; ++wi) {
      samples.getWord(wi).getArgs().resize((unsigned long) p_size);
      for (int pi = 0; pi < p_size; pi++) {
        int pred = getMaxId(as_vector(hg.incremental_forward(adists[pi * w_size + wi])));
        samples.getWord(wi).getArgs()[pi] = dict[ARG].convert(pred);
      }
    }
  }
};


#endif //PROJECT_STNLSTM_H
