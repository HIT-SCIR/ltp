#include "dynet/weight-decay.h"

namespace dynet {

DYNET_SERIALIZE_COMMIT(L2WeightDecay, DYNET_SERIALIZE_DEFINE(weight_decay, lambda))
DYNET_SERIALIZE_IMPL(L2WeightDecay)

}
