//
// Created by liu on 2017/4/10.
//

#ifndef PROJECT_MLPMODELBUILDER_H
#define PROJECT_MLPMODELBUILDER_H

#include "./ModelBuilder.h"
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include <dynet/expr.h>
#include <dynet/model.h>
#include "vector"

using namespace dynet;
using namespace dynet::expr;
using namespace std;

class MLPModelBuilder : public ModelBuilder<Expression, Expression> {
  vector<unsigned> layerDims;
  unsigned outDim;
  vector<Parameter> bias;
  vector<Parameter> mulParams;
public:
  MLPModelBuilder(vector<unsigned> layerDims, unsigned outDim) : layerDims(layerDims), outDim(outDim)
  {
  }

  virtual void init(dynet::Model &model) {
    assert(layerDims.size() > 0);
    assert(outDim > 0);
    for (int i = 0; i < layerDims.size() - 1; ++i) {
      mulParams.push_back(model.add_parameters({layerDims[i + 1], layerDims[i]}));
      bias.push_back(model.add_parameters({layerDims[i + 1]}));
    }
    mulParams.push_back(model.add_parameters({outDim, layerDims[layerDims.size() - 1]}));
    bias.push_back(model.add_parameters({outDim}));
  }

  virtual Expression forward(dynet::ComputationGraph &hg, const Expression& features) {
    Expression hiddenLayer = features;
    for (int i = 0; i < mulParams.size(); ++i) {
      hiddenLayer = (parameter(hg, mulParams[i]) * hiddenLayer + parameter(hg, bias[i]));
      hiddenLayer = dynet::expr::logistic(hiddenLayer);
    }
    return hiddenLayer;
  }

};

#endif //PROJECT_MLPMODELBUILDER_H
