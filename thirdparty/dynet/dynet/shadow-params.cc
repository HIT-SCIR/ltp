#include "dynet/dynet.h"

#include "dynet/shadow-params.h"
#include "dynet/tensor.h"
#include "dynet/aligned-mem-pool.h"
#include "dynet/model.h"

#define LOAD_INIT_FUNC() initialize_lookups()

using namespace std;

namespace dynet {

ShadowParameters::ShadowParameters(const ParameterStorage& p) : h(p.values) {
  default_device->allocate_tensor(DeviceMempool::PS, h);
  TensorTools::zero(h);
}

ShadowLookupParameters::ShadowLookupParameters(const LookupParameterStorage& lp) : all_h(lp.all_values) {
  default_device->allocate_tensor(DeviceMempool::PS, all_h);
  TensorTools::zero(all_h);
  initialize_lookups();
}

void ShadowLookupParameters::initialize_lookups() {
  int num = all_h.d[all_h.d.nd-1];
  Dim dim = all_h.d; dim.nd--;
  int dim_size = dim.size();
  if(h.size() == 0) {
    h.resize(num);
    for(int i = 0; i < num; ++i)
      h[i] = Tensor(dim, all_h.v + i*dim_size, all_h.device, all_h.mem_pool);
  }
}

vector<ShadowParameters> allocate_shadow_parameters(const Model& m) {
  vector<ShadowParameters> v;
  v.reserve(m.parameters_list().size());
  for (auto& p : m.parameters_list())
    v.emplace_back(*p);
  return v;
}

vector<ShadowLookupParameters> allocate_shadow_lookup_parameters(const Model& m) {
  vector<ShadowLookupParameters> v;
  v.reserve(m.lookup_parameters_list().size());
  for (auto& p : m.lookup_parameters_list())
    v.emplace_back(*p);
  return v;
}

DYNET_SERIALIZE_COMMIT(ShadowParameters, DYNET_SERIALIZE_DEFINE(h))
DYNET_SERIALIZE_IMPL(ShadowParameters)

DYNET_SERIALIZE_SAVE_COMMIT(ShadowLookupParameters, DYNET_SERIALIZE_DEFINE(h))
DYNET_SERIALIZE_LOAD_COMMIT(ShadowLookupParameters, LOAD_INIT_FUNC(), DYNET_SERIALIZE_DEFINE(h))
DYNET_SAVELOAD_IMPL(ShadowLookupParameters)

} // namespace dynet

