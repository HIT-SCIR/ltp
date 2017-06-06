#include "dynet/grad-check.h"

#include <iostream>
#include <algorithm>

#include "dynet/model.h"
#include "dynet/dynet.h"
#include "dynet/tensor.h"
#include "dynet/expr.h"

using namespace std;

namespace dynet {

bool check_grad(Model& m, expr::Expression& expr, int verbosity) {
  ComputationGraph& g = *expr.pg;
  // Clear the parameters first
  const vector<ParameterStorage*>& params = m.parameters_list();
  const vector<LookupParameterStorage*>& lookup_params = m.lookup_parameters_list();
  for (auto pp : params)
    pp->clear();
  for (auto pp : lookup_params)
    pp->clear();

  // Perform forward and backward steps
  float alpha = 5e-4;
  g.forward(expr);
  g.backward(expr);

  // Check
  bool flag = false, curr_flag = false;
  for (auto pp : params) {
    if(verbosity > 1)
      cerr << endl << "PARAMETERS " << pp << endl;
    ParameterStorage& p = *pp;
    if(p.g.v == nullptr) continue;
    size_t ts = p.dim.size();
    for (size_t i = 0; i < ts; ++i) {
      float old = TensorTools::access_element(p.values, i);
      TensorTools::set_element(p.values, i, old - alpha);
      float E_left = as_scalar(g.forward(expr));
      TensorTools::set_element(p.values, i, old + alpha);
      float E_right = as_scalar(g.forward(expr));
      TensorTools::set_element(p.values, i, old);
      float g = (E_right - E_left) / (2 * alpha);
      float g_act = TensorTools::access_element(p.g, i);
      float f = fabs(g - g_act);
      float m = std::max(fabs(g), fabs(g_act));
      if (f > 0.01 && m > 0.f) f /= m;
      if (f > 0.01 || std::isnan(f)) { flag = true; if(verbosity > 0) { curr_flag = true; cerr << "***[" << f << "] "; } }
      if(verbosity + (curr_flag ? 1 : 0) > 1) {
        cerr << g_act << ' ' << g << endl;
        curr_flag = false;
      }
    }
  }

  for (auto pp : lookup_params) {
    if(verbosity > 1)
      cerr << endl << "LOOKUP PARAMETERS " << pp << endl;
    LookupParameterStorage& p = *pp;
    size_t ts = p.dim.size();
    for (unsigned j : p.non_zero_grads) {
      if(verbosity > 1)
        cerr << "OBJECT=" << j << endl;
      Tensor& v = p.values[j];
      Tensor& ag = p.grads[j];
      for (size_t i = 0; i < ts; ++i) {
        float old = TensorTools::access_element(v, i);
        TensorTools::set_element(v, i, old - alpha);
        float E_left = as_scalar(g.forward(expr));
        TensorTools::set_element(v, i, old + alpha);
        float E_right = as_scalar(g.forward(expr));
        TensorTools::set_element(v, i, old);
        float g = (E_right - E_left) / (2 * alpha);
        float g_act = TensorTools::access_element(ag, i);
        float f = fabs(g - g_act);
        float m = std::max(fabs(g), fabs(g_act));
        if (f > 0.01 && m > 0.f) f /= m;
        if (f > 0.01 || std::isnan(f)) { flag = true; if(verbosity > 0) { curr_flag = true; cerr << "***[" << f << "] "; } }
        if(verbosity + (curr_flag ? 1 : 0) > 1) {
          cerr << g_act << ' ' << g << endl;
          curr_flag = false;
        }
      }
    }
  }

  if (flag) {
    if (verbosity > 1)
      cerr << endl << "*** GRADIENT CHECK FAILED ***" << endl;
  } else {
    if (verbosity > 0)
      cerr << endl << "GRADIENT CHECK PASSED" << endl;
  }
  return !flag;
}

}

