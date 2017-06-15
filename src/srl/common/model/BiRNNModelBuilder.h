//
// Created by liu on 2017/5/22.
//

#ifndef BILSTM_SRL_BIRNNMODELBUILDER_H
#define BILSTM_SRL_BIRNNMODELBUILDER_H

#include "./ModelBuilder.h"
#include "./RNNModelBuilder.h"
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

template<class DynetRnnBuilder> class BiRNNModelBuilder;
typedef BiRNNModelBuilder<LSTMBuilder> BiLSTMModelBuilder;
typedef BiRNNModelBuilder<GRUBuilder> BiGRUModelBuilder;
typedef BiRNNModelBuilder<SimpleRNNBuilder> BiSimpleRNNModelBuilder;

template<class DynetRnnBuilder>
class BiRNNModelBuilder  : public ModelBuilder<vector<Expression>, vector<Expression>> {
  unsigned layers;
  unsigned inputDim;
  unsigned outputDim;
  RNNModelBuilder<DynetRnnBuilder> forwardRNN;
  RNNModelBuilder<DynetRnnBuilder> backwardRNN;
  Parameter begin;
  Parameter end;
public:
  BiRNNModelBuilder(unsigned layers = 0, unsigned inputDim = 0, unsigned outputDim = 0):
          layers(layers),
          inputDim(inputDim),
          outputDim(outputDim),
          forwardRNN(layers, inputDim, outputDim/2),
          backwardRNN(layers, inputDim, outputDim/2)
  {
    assert(outputDim % 2 == 0);
  }

  void setLayers(unsigned int layers) {
    BiRNNModelBuilder::layers = layers;
    forwardRNN.setLayers(layers);
    backwardRNN.setLayers(layers);
  }

  void setInputDim(unsigned int inputDim) {
    BiRNNModelBuilder::inputDim = inputDim;
    forwardRNN.setInputDim(inputDim);
    backwardRNN.setInputDim(inputDim);
  }

  void setOutputDim(unsigned int outputDim) {
    BiRNNModelBuilder::outputDim = outputDim;
    forwardRNN.setOutputDim(outputDim);
    backwardRNN.setOutputDim(outputDim);
  }

  virtual void init(Model &model) {
    assert(layers > 0);
    assert(inputDim > 0);
    assert(outputDim > 0);
    begin = model.add_parameters({inputDim});
    end = model.add_parameters({inputDim});
    forwardRNN.init(model, begin, end);
    backwardRNN.init(model, begin, end); // use backward function will handle this.
  }

  void dropOut(float d) {
    forwardRNN.dropOut(d);
    backwardRNN.dropOut(d);
  }

  void disableDropOut() {
    forwardRNN.disableDropOut();
    backwardRNN.disableDropOut();
  }

  void newGraph(ComputationGraph &cg) {
    forwardRNN.newGraph(cg);
    backwardRNN.newGraph(cg);
  }
  void startNewSequence(vector<Expression> h_0 = {}) {
    forwardRNN.startNewSequence(h_0);
    backwardRNN.startNewSequence(h_0);
  }
  /**
   *
   * @param hg
   * @param aClass
   * @return A 2*outputDim Dim Expression
   */
  virtual vector<Expression> forward(dynet::ComputationGraph &hg, const vector<Expression>& inputList) {
    vector<Expression> res;
    vector<Expression> fw = forwardRNN.forward(hg, inputList);
    vector<Expression> bw = backwardRNN.backward(hg, inputList);
    int size = (int) fw.size();
    for (int i = 0; i < size; ++i) {
      res.push_back(concatenate({fw[i], bw[i]}));
    }
    return res;
  }

  virtual Expression forwardBack(dynet::ComputationGraph &hg, vector<Expression> inputList) {
    return concatenate({
                               forwardRNN.forwardBack(hg, inputList),
                               backwardRNN.backwardBack(hg, inputList)
                       });
  }

  virtual Expression forwardBy2Order(dynet::ComputationGraph &hg, vector<Expression> inputList, vector<int> order1, vector<int> order2) {
    return concatenate({
                               forwardRNN.forwardByOrder(hg, inputList, order1),
                               backwardRNN.forwardByOrder(hg, inputList, order2)
                       });
  }
  virtual Expression forwardBy2Order(dynet::ComputationGraph &hg, vector<Expression> inputList, vector<int> order1, vector<int> order2, Expression& escape) {
    return concatenate({
                               forwardRNN.forwardByOrder(hg, inputList, order1, escape),
                               backwardRNN.forwardByOrder(hg, inputList, order2, escape)
                       });
  }

  virtual Expression forwardBackBy2Order(dynet::ComputationGraph &hg, vector<Expression> inputList, vector<int> order1, vector<int> order2) {
    return concatenate({
                               forwardRNN.forwardBackByOrder(hg, inputList, order1),
                               backwardRNN.forwardBackByOrder(hg, inputList, order2)
                       });
  }
  virtual Expression forwardBackBy2Order(dynet::ComputationGraph &hg, vector<Expression> inputList, vector<int> order1, vector<int> order2, Expression& escape) {
    return concatenate({
                               forwardRNN.forwardBackByOrder(hg, inputList, order1, escape),
                               backwardRNN.forwardBackByOrder(hg, inputList, order2, escape)
                       });
  }

  virtual Expression forwardBackBy2Path(dynet::ComputationGraph &hg, vector<Expression> fwPath, vector<Expression> bwPath) {
    return concatenate({
                               forwardRNN.forwardBack(hg, fwPath),
                               backwardRNN.forwardBack(hg, bwPath)
                       });
  }
};


#endif //BILSTM_SRL_BIRNNMODELBUILDER_H
