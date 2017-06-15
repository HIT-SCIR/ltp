//
// Created by liu on 2017/4/7.
//

#ifndef PROJECT_CNNLAYERBUILDER_H
#define PROJECT_CNNLAYERBUILDER_H

#include "ModelBuilder.h"
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
 * CNN 卷积+pooling层构造器
 * 从Expression数组（一组图像）到Expression数组（另一组图像（卷积+pooling之后））
 *
 * in_rows 输入序列每个元素的向量维度
 * k_fold 1 no folding, 2 fold two rows together, 3 ... 折叠在一起的目的是一起kmax_pooling
 * filter_width 卷积核宽度
 * in_nfmaps 输入张数
 * out_nfmaps 输出张数
 * out_length 输出长度 （输出宽度=in_rows/k_fold）
 */
class CNN1dLayerBuilder: public ModelBuilder<vector<Expression>, vector<Expression>> {
protected:
  int in_rows, k_fold_rows, filter_width, in_nfmaps, out_nfmaps, out_length;

  vector<vector<Parameter>> p_filts; // [feature map index from][feature map index to]
  vector<vector<Parameter>> p_fbias; // [feature map index from][feature map index to]
public:
  CNN1dLayerBuilder(int in_rows, int k_fold_rows, int filter_width, int in_nfmaps, int out_nfmaps, int out_length);

  virtual void init(dynet::Model &model);

  virtual vector<Expression> forward(dynet::ComputationGraph &hg, const vector<Expression> &aClass);

};


#endif //PROJECT_CNNLAYERBUILDER_H
