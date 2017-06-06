#ifndef DYNET_NODES_CONTRACT_H_
#define DYNET_NODES_CONTRACT_H_

#include "dynet/dynet.h"
#include "dynet/devices.h"
#include "dynet/nodes-macros.h"

// See nodes-macros.h for more details about DYNET_NODE_DEFINE_DEV_IMPL().

namespace dynet {

// Forward:
//   Y_ij = A_ijk * B_k + C_ij
//
// Backward:
//   (dE/dA)_ijk = (dE/dY)_ij * L_k
//   (dE/dB)_k = (dE/dY)_ij * A_ijk
//   (dE/dC)_ij = (dE/dY)_ij
struct InnerProduct3D_1D : public Node {
  InnerProduct3D_1D(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

//   Y_i = A_ijk * B_k * C_j
struct InnerProduct3D_1D_1D : public Node {
  InnerProduct3D_1D_1D(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  DYNET_NODE_DEFINE_DEV_IMPL()
};

} // namespace dynet

#endif
