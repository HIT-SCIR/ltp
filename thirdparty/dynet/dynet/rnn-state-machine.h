#ifndef DYNET_RNN_STATE_MACHINE_H_
#define DYNET_RNN_STATE_MACHINE_H_

#include "dynet/io-macros.h"

namespace dynet {

// CURRENT STATE | ACTION              | NEXT STATE
// --------------+---------------------+-----------------
// CREATED       | new_graph           | GRAPH_READY
// GRAPH_READY   | start_new_sequence  | READING_INPUT
// READING_INPUT | add_input           | READING_INPUT
// READING_INPUT | start_new_seqeunce  | READING_INPUT
// READING_INPUT | new_graph           | GRAPH_READY

enum RNNState {CREATED, GRAPH_READY, READING_INPUT};
enum RNNOp {new_graph, start_new_sequence, add_input};

class RNNStateMachine {
 public:
  RNNStateMachine() : q_(RNNState::CREATED) {}
  void failure(RNNOp op);
  void transition(RNNOp op) {
    switch (q_) {
      case RNNState::CREATED:
        if (op == RNNOp::new_graph) { q_ = RNNState::GRAPH_READY; break; }
        failure(op);
      case RNNState::GRAPH_READY:
        if (op == RNNOp::new_graph) { break; }
        if (op == RNNOp::start_new_sequence) { q_ = RNNState::READING_INPUT; break; }
        failure(op);
      case RNNState::READING_INPUT:
        if (op == RNNOp::add_input) { break; }
        if (op == RNNOp::start_new_sequence) { break; }
        if (op == RNNOp::new_graph) { q_ = RNNState::GRAPH_READY; break; }
        failure(op);
    }
  }
 private:
  RNNState q_;

  DYNET_SERIALIZE_DECLARE()
};

} // namespace dynet

#endif
