//
// Created by liu on 2017/2/20.
//

#ifndef TTPLAB_AFFINETRANSFORMMODELBUILDER_H
#define TTPLAB_AFFINETRANSFORMMODELBUILDER_H

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

/**
 * 输出表达式 = （输入表达式 * 类内参数）+ 偏移
 * 注意顺序
 * 对象命名含有顺序
 */
class AffineTransformModelBuilder: public ModelBuilder<vector<Expression>, Expression>  {
  vector<unsigned> inputDims;
  unsigned outDim;
  Parameter bias;
  vector<Parameter> mulParams;
public:
  AffineTransformModelBuilder(vector<unsigned> inputDims = {}, unsigned outDim = 0):
          inputDims(inputDims),
          outDim(outDim)
  { }

  void setInputDims(const vector<unsigned int> &inputDims) {
    AffineTransformModelBuilder::inputDims = inputDims;
  }

  void setOutDim(unsigned int outDim) {
    AffineTransformModelBuilder::outDim = outDim;
  }

  virtual void init(dynet::Model &model) {
    assert(inputDims.size() > 0);
    assert(outDim > 0);
    bias = model.add_parameters({outDim});
    for (int i = 0; i < inputDims.size(); ++i) {
      mulParams.push_back(model.add_parameters({outDim, inputDims[i]}));
    }
  }

  virtual Expression forward(dynet::ComputationGraph &hg, const vector<Expression>& features);

  virtual vector<float> _debug_get_para(dynet::ComputationGraph &hg) {
     return as_vector(hg.incremental_forward(parameter(hg, bias)));
  }

};


#endif //TTPLAB_AFFINETRANSFORMMODELBUILDER_H
