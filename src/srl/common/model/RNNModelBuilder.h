//
// Created by liu on 2017/5/22.
//

#ifndef BILSTM_SRL_RNNMODELBUILDER_H
#define BILSTM_SRL_RNNMODELBUILDER_H

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
#include <vector>

using namespace dynet;
using namespace dynet::expr;
using namespace std;

template <class DynetRnnBuilder> class RNNModelBuilder;
typedef RNNModelBuilder<LSTMBuilder> LSTMModelBuilder;
typedef RNNModelBuilder<GRUBuilder> GRUModelBuilder;
typedef RNNModelBuilder<SimpleRNNBuilder> SimpleRNNModelBuilder;

template <class DynetRnnBuilder>
class RNNModelBuilder : public ModelBuilder<vector<Expression>, vector<Expression>>{
  unsigned layers;
  unsigned inputDim;
  unsigned outputDim;
  DynetRnnBuilder dynetRnnBuilder;
  Parameter begin;
  Parameter end;
public:
  RNNModelBuilder(unsigned layers, unsigned inputDim, unsigned outputDim) :
          layers(layers),
          inputDim(inputDim),
          outputDim(outputDim)
  {}

  void setLayers(unsigned int _layers) { layers = _layers; }

  void setInputDim(unsigned int _inputDim) { inputDim = _inputDim; }

  void setOutputDim(unsigned int _outputDim) { outputDim = _outputDim; }

  using ModelBuilder<vector<Expression>, vector<Expression>>::init;
  virtual void init(Model &model, bool initBeginEnd = true) {
    dynetRnnBuilder = DynetRnnBuilder(layers, inputDim, outputDim, model);
    if (initBeginEnd) {
      begin = model.add_parameters({inputDim});
      end = model.add_parameters({inputDim});
    }
  }

  virtual void init(Model &model, Parameter& begin, Parameter& end) {
    this->begin = begin;
    this->end = end;
    init(model, false);
  }

  void newGraph(ComputationGraph &cg) { dynetRnnBuilder.new_graph(cg); }
  void startNewSequence(vector<Expression> h_0 = {}) { dynetRnnBuilder.start_new_sequence(); }

  void dropOut(float d) { if (d > 1e-6) { dynetRnnBuilder.set_dropout(d); } else { dynetRnnBuilder.disable_dropout(); } }

  void disableDropOut() { dynetRnnBuilder.disable_dropout(); }

  /**
   *
   * @param hg
   * @param inputList
   * @return
   *
   * begin -> [1 -> 2 -> ... -> last] -> end
   *           |    |   |...|    |
   *           V    V   V   V    V
   *          [1 -> 2 -> ... -> last] return this out put
   */
  virtual vector<Expression> forward(dynet::ComputationGraph &hg, const vector<Expression>& inputList) {
    vector<Expression> res;
    dynetRnnBuilder.add_input(parameter(hg, begin));
    for (int i = 0; i < inputList.size(); ++i) {
      res.push_back(dynetRnnBuilder.add_input(inputList[i]));
    }
    dynetRnnBuilder.add_input(parameter(hg, end));
    return res;
  }

  /**
   * this is used for backward lstm in BiLSTM
   * note : use forward or use backward only! Use both is logically wrong!
   * @param hg
   * @param inputList
   * @return
   *
   * begin <- [1 <- 2 <- ... <- last] <- end
   *           |    |   |...|    |
   *           V    V   V   V    V
   *          [1 -> 2 -> ... -> last] return this out put
   */

  virtual vector<Expression> backward(dynet::ComputationGraph &hg, const vector<Expression>& inputList) {
    vector<Expression> res(inputList.size());
    dynetRnnBuilder.add_input(parameter(hg, end));
    for (int i = inputList.size() - 1; i >= 0; --i) {
      res[i] = dynetRnnBuilder.add_input(inputList[i]);
    }
    dynetRnnBuilder.add_input(parameter(hg, begin));
    return res;
  }

  /**
   *
   * @param hg
   * @param inputList
   * @return
   *
   * begin -> [1 -> 2 -> ... -> last] -> end
   *                                     |
   *                                     V
   *                                    EXP return this out put
   */
  virtual Expression forwardBack(dynet::ComputationGraph &hg, vector<Expression>& inputList) {
    dynetRnnBuilder.add_input(parameter(hg, begin));
    for (int i = 0; i < inputList.size(); ++i) {
      dynetRnnBuilder.add_input(inputList[i]);
    }
    dynetRnnBuilder.add_input(parameter(hg, end));
    return dynetRnnBuilder.back();
  }

  /**
   *
   * @param hg
   * @param inputList
   * @return
   *
   * begin <- [1 <- 2 <- ... <- last] <- end
   *   |
   *   V
   *  EXP return this out put
   */
  virtual Expression backwardBack(dynet::ComputationGraph &hg, vector<Expression>& inputList) {
    dynetRnnBuilder.add_input(parameter(hg, end));
    for (int i = inputList.size() - 1; i >= 0; --i) {
      dynetRnnBuilder.add_input(inputList[i]);
    }
    dynetRnnBuilder.add_input(parameter(hg, begin));
    return dynetRnnBuilder.back();
  }

  /**
   *
   * @param hg
   * @param inputList
   * @param order
   * @return
   *
   * begin -> [1 -> 2 -> ... -> last] -> end
   *                              |
   *                              V
   *                             EXP return this out put
   */
  virtual Expression forwardByOrder(dynet::ComputationGraph &hg, vector<Expression>& inputList, vector<int>& order) {
    dynetRnnBuilder.add_input(parameter(hg, begin));
    for (int i = 0; i < order.size(); ++i) {
      dynetRnnBuilder.add_input(inputList[order[i]]);
    }
    return dynetRnnBuilder.back();
  }

  virtual Expression forwardByOrder(dynet::ComputationGraph &hg, vector<Expression>& inputList, vector<int>& order, Expression& escape) {
    dynetRnnBuilder.add_input(parameter(hg, begin));
    for (int i = 0; i < order.size(); ++i) {
      dynetRnnBuilder.add_input(order[i] > 0 ? inputList[order[i]] : escape);
    }
    return dynetRnnBuilder.back();
  }
  /**
    *
    * @param hg
    * @param inputList
    * @param order
    * @return
    *
    * begin -> [1 -> 2 -> ... -> last] -> end
    *                                      |
    *                                      V
    *                                     EXP return this out put
    */
  virtual Expression forwardBackByOrder(dynet::ComputationGraph &hg, vector<Expression>& inputList, vector<int>& order) {
    dynetRnnBuilder.add_input(parameter(hg, begin));
    for (int i = 0; i < order.size(); ++i) {
      dynetRnnBuilder.add_input(inputList[order[i]]);
    }
    dynetRnnBuilder.add_input(parameter(hg, end));
    return dynetRnnBuilder.back();
  }
  virtual Expression forwardBackByOrder(dynet::ComputationGraph &hg, vector<Expression>& inputList, vector<int>& order, Expression& escape) {
    dynetRnnBuilder.add_input(parameter(hg, begin));
    for (int i = 0; i < order.size(); ++i) {
      dynetRnnBuilder.add_input(order[i] > 0 ? inputList[order[i]] : escape);
    }
    dynetRnnBuilder.add_input(parameter(hg, end));
    return dynetRnnBuilder.back();
  }

};

#endif //BILSTM_SRL_RNNMODELBUILDER_H
