//
// Created by liu on 2017/5/11.
//

#include "AffineTransformModelBuilder.h"


Expression AffineTransformModelBuilder::forward(dynet::ComputationGraph &hg, const vector<Expression>& features) {
  assert(features.size() == mulParams.size());
  vector<Expression> tranParams;
  tranParams.push_back(parameter(hg, bias));
  for (int i = 0; i < mulParams.size(); ++i) {
    tranParams.push_back(parameter(hg, mulParams[i]));
    tranParams.push_back(features[i]);
  }
  return affine_transform(tranParams);
}