#ifndef DYNET_WEIGHT_DECAY_H
#define DYNET_WEIGHT_DECAY_H

#include <stdexcept>
#include <cmath>
#include <iostream>
#include "dynet/io-macros.h"

namespace dynet {

// I don't bother with learning rates when computing how much the weight
// decay changes- those are hard to define in the adaptive update rules.
// So, we do something simple that works with everything.
//
// Note: you may want to discount lambda as you learn if your eta is on a
// decaying schedule.
struct L2WeightDecay {
  explicit L2WeightDecay(float lambda = 1e-6) : weight_decay(1) { set_lambda(lambda); }
  void set_lambda(float lam) {
    if (lam < 0) throw std::domain_error("Bad value of lambda in set_lambda");
    lambda = lam;
  }
  void update_weight_decay(unsigned num_updates = 1) {
    if (num_updates == 0) return;
    if (num_updates == 1)
      weight_decay -= weight_decay * lambda;
    else weight_decay = weight_decay * std::pow(1-lambda, num_updates);
  }
  float current_weight_decay() const { return weight_decay; }
  bool parameters_need_rescaled() const {
    return (weight_decay < 0.25f);
  }
  void reset_weight_decay() {
    std::cerr << "RESCALE WEIGHT DECAY FROM " << weight_decay << " to 1.0\n";
    weight_decay = 1.0f;
  }
 private:
  DYNET_SERIALIZE_DECLARE()

  float weight_decay;
  float lambda;
};

} // namespace dynet

#endif
