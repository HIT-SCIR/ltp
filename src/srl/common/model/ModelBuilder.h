//
// Created by liu on 2017/2/20.
//

#ifndef TTPLAB_MODELBUILDER_H
#define TTPLAB_MODELBUILDER_H

#include <dynet/model.h>

/**
 * Builder
 * 此类和派生类的作用是辅助model类搭建复杂模型，实际上是一个参数分组类的接口类。
 *
 * 一个builder应该包含
 * - 从某种表达式（或数字）到目标表达式的推导方法。
 * - 并且包含运算过程相关参数。
 *
 * 一个builder可以被其他builder组合包含
 *
 * 每个builder必须实现此类接口
 */
template <class InputClass, class OutputClass>
class ModelBuilder {
public:
  dynet::Model * model;
  ModelBuilder() {

  }

  virtual void init(dynet::Model & model) {
    this->model = & model;
  }


  virtual OutputClass forward(dynet::ComputationGraph &hg, const InputClass &) {
    return OutputClass();
  };

  virtual OutputClass forward(dynet::ComputationGraph &hg, const InputClass && in) {
    return forward(hg, in);
  }

};


#endif //TTPLAB_MODELBUILDER_H
