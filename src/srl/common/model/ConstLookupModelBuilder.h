//
// Created by liu on 2017/3/9.
//

#ifndef PROJECT_CONSTLOOKUPMODELBUILDER_H
#define PROJECT_CONSTLOOKUPMODELBUILDER_H

#include "LookupModelBuilder.h"
#include <dynet/expr.h>

class ConstLookupModelBuilder: public LookupModelBuilder {
public:
  ConstLookupModelBuilder(unsigned inputDim = 0, unsigned expressionDim = 0):
          LookupModelBuilder(inputDim, expressionDim) { }

  virtual Expression forward(ComputationGraph &hg, const unsigned & num) {
    return dynet::expr::const_lookup(hg, lookupParameter, num);
  }

};


#endif //PROJECT_CONSTLOOKUPMODELBUILDER_H
