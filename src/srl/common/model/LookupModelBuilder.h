//
// Created by liu on 2017/2/20.
//

#ifndef TTPLAB_LOOKUPMODELBUILDER_H
#define TTPLAB_LOOKUPMODELBUILDER_H

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

class LookupModelBuilder: public ModelBuilder<unsigned, Expression> {
protected:
  unsigned inputDim, expressionDim;
  LookupParameter lookupParameter;
public:
  LookupModelBuilder(unsigned inputDim = 0, unsigned expressionDim = 0):
          inputDim(inputDim),
          expressionDim(expressionDim) { }

  unsigned getExpressionDim() const {
    return expressionDim;
  }

  unsigned getInputDim() const {
    return inputDim;
  }

  void setInputDim(unsigned int inputDim) {
    LookupModelBuilder::inputDim = inputDim;
  }

  void setExpressionDim(unsigned int expressionDim) {
    LookupModelBuilder::expressionDim = expressionDim;
  }

  virtual void init(Model & model) {
    assert(inputDim > 0);
    assert(expressionDim > 0);
    lookupParameter = model.add_lookup_parameters(inputDim, {expressionDim});
  }

  virtual Expression forward(ComputationGraph &hg, const unsigned & num) {
    return lookup(hg, lookupParameter, num);
  }

  virtual vector<Expression> forwardList(ComputationGraph &hg, const vector<unsigned> & nums) {
    vector<Expression> res;
    for (int i = 0; i < nums.size(); i++) {
      res.push_back(forward(hg, nums[i]));
    }
    return res;
  }

  void initialize(unsigned index, vector<float>& val) {
    lookupParameter.initialize(index, val);
  }

};


#endif //TTPLAB_LOOKUPMODELBUILDER_H
