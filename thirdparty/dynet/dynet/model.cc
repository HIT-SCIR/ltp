#include "dynet/model.h"
#include "dynet/tensor.h"
#include "dynet/aligned-mem-pool.h"
#include "dynet/dynet.h"

#include <unordered_set>
#include <iostream>

#include <fstream>
#include <sstream>


#include <stdexcept>

#define LOAD_INIT_FUNC() initialize_lookups()

#ifdef __CUDACC__
#include "dynet/gpu-ops.h"
#endif

// Macros for defining functions over parameters
// NOTE: This only works on the default device, as parameters are currently defined over default devices
#ifdef __CUDACC__
#define DYNET_PARAMNORM_INST_DEV_IMPL(MyParam, regular_func, dev_func) \
  template void MyParam::dev_func<Device_GPU>(Device_GPU & dev, float *sqnorm) const;
#elif defined(HAVE_CUDA)
#define DYNET_PARAMNORM_INST_DEV_IMPL(MyParam, regular_func, dev_func) \
  extern template void MyParam::dev_func<Device_GPU>(Device_GPU & dev, float *sqnorm) const; \
  template void MyParam::dev_func<Device_CPU>(Device_CPU & dev, float *sqnorm) const; \
  void MyParam::regular_func(float *sqnorm) const { \
    if(default_device->type == DeviceType::CPU) { dev_func(*(Device_CPU*)default_device,sqnorm); } \
    else if(default_device->type == DeviceType::GPU) { dev_func(*(Device_GPU*)default_device,sqnorm); } \
    else { throw std::runtime_error("Invalid device type in MyParam::dev_func"); } \
  }
#else
#define DYNET_PARAMNORM_INST_DEV_IMPL(MyParam, regular_func, dev_func) \
  template void MyParam::dev_func<Device_CPU>(Device_CPU & dev, float *sqnorm) const; \
  void MyParam::regular_func(float *sqnorm) const { \
    if(default_device->type == DeviceType::CPU) { dev_func(*(Device_CPU*)default_device,sqnorm); } \
    else { throw std::runtime_error("Invalid device type in MyParam::dev_func"); } \
  }
#endif

using namespace std;

namespace dynet {

// CPU only functions
#ifndef __CUDACC__

ParameterStorageBase::~ParameterStorageBase() {}
DYNET_SERIALIZE_IMPL(ParameterStorageBase)

ParameterStorage::ParameterStorage(const Dim& d, float scale) : dim(d) {
  values.d = g.d = d;
  values.device = g.device = default_device;
  default_device->allocate_tensor(DeviceMempool::PS, values);
  default_device->allocate_tensor(DeviceMempool::PS, g);
  TensorTools::zero(g);
  if (scale == 0.0f) {
    ParameterInitGlorot init;
    init.initialize_params(values);
  } else {
    ParameterInitUniform init(scale);
    init.initialize_params(values);
  }
}

ParameterStorage::ParameterStorage(const Dim& d, const ParameterInit & init) : dim(d) {
  values.d = g.d = d;
  values.device = g.device = default_device;
  default_device->allocate_tensor(DeviceMempool::PS, values);
  default_device->allocate_tensor(DeviceMempool::PS, g);
  TensorTools::zero(g);
  init.initialize_params(values);
}

size_t ParameterStorage::size() const { return dim.size(); }

void ParameterStorage::zero() {
  TensorTools::zero(values);
  clear();
}

void ParameterStorage::copy(const ParameterStorage & param) {
  DYNET_ARG_CHECK(dim == param.dim,
                          "Attempt to copy between parameters with mismatched dimensions: " << dim << " != " << param.dim);
  TensorTools::copy_elements(values, param.values);
}

void ParameterStorage::clear() {
  if (g.v != nullptr)
    TensorTools::zero(g);
}

void ParameterStorage::clip(float left, float right) {
  TensorTools::clip(values, left, right);
}

#ifndef __CUDACC__
DYNET_SERIALIZE_COMMIT(ParameterStorage,
                       DYNET_SERIALIZE_DERIVED_DEFINE(ParameterStorageBase, dim, values, g))
DYNET_SERIALIZE_IMPL(ParameterStorage)
#endif

LookupParameterStorage::LookupParameterStorage(unsigned n, const Dim& d) : dim(d), all_updated(false) {
  all_dim = dim; all_dim.d[all_dim.nd++] = n;
  all_grads.d = all_values.d = all_dim;
  all_grads.device = all_values.device = default_device;
  default_device->allocate_tensor(DeviceMempool::PS, all_values);
  default_device->allocate_tensor(DeviceMempool::PS, all_grads);
  ParameterInitGlorot init(true);
  init.initialize_params(all_values);
  initialize_lookups();
}

LookupParameterStorage::LookupParameterStorage(unsigned n, const Dim& d, const ParameterInit & init) : dim(d), all_updated(false) {
  all_dim = dim; all_dim.d[all_dim.nd++] = n;
  all_grads.d = all_values.d = all_dim;
  all_grads.device = all_values.device = default_device;
  default_device->allocate_tensor(DeviceMempool::PS, all_values);
  default_device->allocate_tensor(DeviceMempool::PS, all_grads);
  init.initialize_params(all_values);
  initialize_lookups();
}

void LookupParameterStorage::initialize_lookups() {
  int num = all_dim[all_dim.nd - 1];
  dim = all_dim; dim.nd--;
  int dim_size = dim.size();
  if (values.size() == 0) {
    values.resize(num);
    for (int i = 0; i < num; ++i)
      values[i] = Tensor(dim, all_values.v + i * dim_size, all_values.device, all_values.mem_pool);
  }
  if (grads.size() == 0 && all_grads.v != nullptr) {
    grads.resize(num);
    for (int i = 0; i < num; ++i)
      grads[i] = Tensor(dim, all_grads.v + i * dim_size, all_grads.device, all_grads.mem_pool);
  }
}

void LookupParameterStorage::zero() {
  TensorTools::zero(all_values);
}

size_t LookupParameterStorage::size() const {
  return all_dim.size();
}

void LookupParameterStorage::copy(const LookupParameterStorage& param) {
  if(all_dim != param.all_dim)
    DYNET_INVALID_ARG("Attempt to copy between lookup parameters with mismatched dimensions: " << all_dim << " != " << param.all_dim);
  TensorTools::copy_elements(all_values, param.all_values);
}

void LookupParameterStorage::clear() {
  // TODO: the GPU part is hacky, probably need a better heuristic
  if (all_grads.device->type == DeviceType::GPU || all_updated) {
    TensorTools::zero(all_grads);
  } else {
    for (auto i : non_zero_grads)
      TensorTools::zero(grads[i]);
  }
  non_zero_grads.clear();
  all_updated = false;
}

#ifndef __CUDACC__
DYNET_SERIALIZE_SAVE_COMMIT(LookupParameterStorage,
		            DYNET_SERIALIZE_DERIVED_DEFINE(ParameterStorageBase, all_dim, all_values, all_grads))
DYNET_SERIALIZE_LOAD_COMMIT(LookupParameterStorage, LOAD_INIT_FUNC(),
		            DYNET_SERIALIZE_DERIVED_DEFINE(ParameterStorageBase, all_dim, all_values, all_grads))
DYNET_SAVELOAD_IMPL(LookupParameterStorage)
#endif

void ParameterInitNormal::initialize_params(Tensor & values) const {
  TensorTools::randomize_normal(values, mean, sqrt(var));
}

void ParameterInitUniform::initialize_params(Tensor & values) const {
  TensorTools::randomize_uniform(values, left, right);
}

void ParameterInitConst::initialize_params(Tensor & values) const {
  TensorTools::constant(values, cnst);
}

void ParameterInitIdentity::initialize_params(Tensor & values) const {
  TensorTools::identity(values);
}

void ParameterInitGlorot::initialize_params(Tensor & values) const {
  int dims = 0, dim_len = values.d.nd - (lookup ? 1 : 0);
  for (int i = 0; i < dim_len; ++i) dims += values.d[i];
  float my_scale = gain * sqrt(6) / sqrt(dims);
  TensorTools::randomize_uniform(values, -my_scale, my_scale);
}

void ParameterInitSaxe::initialize_params(Tensor & values) const {
  if (values.device->type == DeviceType::GPU)
    throw std::runtime_error("Saxe initialization not implemented for CUDA (we welcome pull requests)");
  else
    TensorTools::randomize_orthonormal(values, gain);
}


void ParameterInitFromVector::initialize_params(Tensor & values) const {
  TensorTools::set_elements(values, vec);
}

void ParameterInitFromFile::initialize_params(Tensor & values) const {
  ifstream is(filename);
  istream_iterator<float> start(is), end;
  vector<float> param_vector(start, end);
  TensorTools::set_elements(values, param_vector);
}


Parameter::Parameter() {
  mp = nullptr;
  index = 0;
}

Parameter::Parameter(Model* mp, unsigned long index) : mp(mp), index(index) {}

ParameterStorage* Parameter::get() const {
  return mp->parameters_list()[index];
}
void Parameter::clip_inplace(float left, float right){
  float my_scale = 1./ mp->weight_decay.current_weight_decay();
  get()->clip(left * my_scale, right * my_scale);
}
void Parameter::zero() {
  return mp->parameters_list()[index]->zero();
}

void Parameter::set_updated(bool b) {
  mp->set_updated_param(this, b);
}

bool Parameter::is_updated() {
  return mp->is_updated_param(this);
}


#ifndef __CUDACC__
DYNET_SERIALIZE_COMMIT(Parameter, DYNET_SERIALIZE_DEFINE(mp, index))
DYNET_SERIALIZE_IMPL(Parameter)
#endif

LookupParameter::LookupParameter() {
  mp = nullptr;
  index = 0;
}

LookupParameter::LookupParameter(Model* mp, unsigned long index) : mp(mp), index(index) {}

LookupParameterStorage* LookupParameter::get() const {
  return mp->lookup_parameters_list()[index];
}

void LookupParameter::zero() {
  return mp->lookup_parameters_list()[index]->zero();
}

void LookupParameter::initialize(unsigned index, const std::vector<float>& val) const {
  get()->initialize(index, val);
}

void LookupParameter::set_updated(bool b) {
  mp->set_updated_lookup_param(this, b);
}
bool LookupParameter::is_updated() {
  return mp->is_updated_lookup_param(this);
}

#ifndef __CUDACC__
DYNET_SERIALIZE_COMMIT(LookupParameter, DYNET_SERIALIZE_DEFINE(mp, index))
DYNET_SERIALIZE_IMPL(LookupParameter)
#endif

Model::Model() : gradient_norm_scratch(nullptr) {
  weight_decay.set_lambda(weight_decay_lambda);
}

Model::~Model() {
  for (auto p : all_params) delete p;
  if (gradient_norm_scratch)
    default_device->mem->free(gradient_norm_scratch);
}

void Model::set_weight_decay_lambda(float lambda) {
  weight_decay.set_lambda(lambda);
}

void Model::project_weights(float radius) {
  static float* project_scratch = 0;
  if (!project_scratch)
    project_scratch = (float*)default_device->mem->malloc(all_params.size() * sizeof(float));
  int pi = 0;
  for (auto p : all_params) {
    p->squared_l2norm(&project_scratch[pi]);
    ++pi;
  }
  double gg = 0;
  for (int i = 0; i < pi; ++i)
    gg += project_scratch[i];
  cerr << "NORM: " << sqrt(gg) << endl;
}

Parameter Model::add_parameters(const Dim& d, float scale) {
  ParameterStorage* p = new ParameterStorage(d, scale);
  Parameter r(this, params.size());
  //cerr << "Adding parameters with dim " << d << endl;
  all_params.push_back(p);
  params.push_back(p);
  updated_params.push_back(r.index);
  return r;
}

Parameter Model::add_parameters(const Dim& d, const ParameterInit & init) {
  ParameterStorage* p = new ParameterStorage(d, init);
  Parameter r(this, params.size());
  //cerr << "Adding parameters with dim " << d << endl;
  all_params.push_back(p);
  params.push_back(p);
  updated_params.push_back(r.index);
  return r;
}


LookupParameter Model::add_lookup_parameters(unsigned n, const Dim& d) {
  LookupParameterStorage* p = new LookupParameterStorage(n, d);
  LookupParameter r(this, lookup_params.size());
  //cerr << "Adding lookup parameters with dim " << d << " and size " << n << endl;
  all_params.push_back(p);
  lookup_params.push_back(p);
  updated_lookup_params.push_back(r.index);
  return r;
}

LookupParameter Model::add_lookup_parameters(unsigned n, const Dim& d, const ParameterInit & init) {
  LookupParameterStorage* p = new LookupParameterStorage(n, d, init);
  LookupParameter r(this, lookup_params.size());
  //cerr << "Adding lookup parameters with dim " << d << " and size " << n << endl;
  all_params.push_back(p);
  lookup_params.push_back(p);
  updated_lookup_params.push_back(r.index);
  return r;
}

void Model::set_updated_param(const Parameter *p, bool status) {
  unsigned idx = p->index;
  DYNET_ASSERT(idx < params.size(), "Parameter ID " << idx << " is less than parameter size " << params.size());

  auto position = std::find(updated_params.begin(), updated_params.end(), idx);
  if (position == updated_params.end()) {
    if (status) updated_params.push_back(idx);
  } else {
    if (!status) updated_params.erase(position);
  }
}

void Model::set_updated_lookup_param(const LookupParameter *p, bool status) {
  unsigned idx = p->index;
  DYNET_ASSERT(idx < lookup_params.size(), "LookupParameter ID " << idx << " is less than lookup parameter size " << lookup_params.size());

  auto position = std::find(updated_lookup_params.begin(), updated_lookup_params.end(), idx);
  if (position == updated_lookup_params.end()) {
    if (status) updated_lookup_params.push_back(idx);
  } else {
    if (!status) updated_lookup_params.erase(position);
  }
}

bool Model::is_updated_param(const Parameter* p) {
  auto position = std::find(updated_params.begin(), updated_params.end(), p->index);
  return position != updated_params.end();
}

bool Model::is_updated_lookup_param(const LookupParameter* p) {
  auto position = std::find(updated_lookup_params.begin(), updated_lookup_params.end(), p->index);
  return position != updated_lookup_params.end();
}

void Model::reset_gradient() {
  for (auto p : params) { p->clear(); }
  for (auto p : lookup_params) { p->clear(); }
}

size_t Model::parameter_count() const {
  size_t r = 0;
  for (const ParameterStorageBase* param : all_params) {
    r += param->size();
  }
  return r;
}

size_t Model::updated_parameter_count() const {
  size_t r = 0;
  for (const unsigned idx : updated_params) {
    r += params[idx]->size();
  }
  for (const unsigned idx : updated_lookup_params) {
    r += lookup_params[idx]->size();
  }
  return r;
}

#ifndef __CUDACC__
DYNET_SERIALIZE_COMMIT(Model,
                       DYNET_SERIALIZE_DEFINE(all_params, params,
                                              lookup_params, weight_decay,
                                              updated_params, updated_lookup_params))
DYNET_SERIALIZE_IMPL(Model)
#endif

void save_dynet_model(std::string filename, Model* model) {
  std::ofstream out(filename);
  boost::archive::text_oarchive oa(out);
  oa << (*model);
};

void load_dynet_model(std::string filename, Model* model) {
  std::ifstream in(filename);
  boost::archive::text_iarchive ia(in);
  ia >> (*model);
};

#endif

// CPU/GPU code
// TODO: It's a bit annoying to re-implement the CPU/GPU control code for each
//       function, but it's not clear how to handle heterogeneous functions w/
//       macros

// Note: Using DeviceMempool::NONE here because these tensors are not persistent
// and won't be saved so it doesn't matter which mempool they belong to.

// Take the squared norm
template <class MyDevice>
void ParameterStorage::squared_l2norm_dev(MyDevice & dev, float* sqnorm) const {
  Tensor sqnorm_t({1}, sqnorm, &dev, DeviceMempool::NONE);
  sqnorm_t.t<0>().device(*dev.edevice) = values.tvec().square().sum();
}
DYNET_PARAMNORM_INST_DEV_IMPL(ParameterStorage, squared_l2norm, squared_l2norm_dev)

// Take the squared norm of the gradient
template <class MyDevice>
void ParameterStorage::g_squared_l2norm_dev(MyDevice & dev, float* sqnorm) const {
  DYNET_ASSERT(g.v != nullptr, "Cannot take norm of gradient with null parameter");
  Tensor sqnorm_t({1}, sqnorm, &dev, DeviceMempool::NONE);
  sqnorm_t.t<0>().device(*dev.edevice) = g.tvec().square().sum();
}
DYNET_PARAMNORM_INST_DEV_IMPL(ParameterStorage, g_squared_l2norm, g_squared_l2norm_dev)

template <class MyDevice>
void ParameterStorage::accumulate_grad_dev(MyDevice & dev, const Tensor& d) {
  g.tvec().device(*dev.edevice) += d.tvec();
}
#ifdef __CUDACC__
template void ParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, const Tensor& d);
#elif defined(HAVE_CUDA)
extern template void ParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, const Tensor& d);
template void ParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, const Tensor& d);
void ParameterStorage::accumulate_grad(const Tensor& d) {
  if (values.device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)values.device, d); }
  else if (values.device->type == DeviceType::GPU) { accumulate_grad_dev(*(Device_GPU*)values.device, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void ParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, const Tensor& d);
void ParameterStorage::accumulate_grad(const Tensor& d) {
  if (values.device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)values.device, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void ParameterStorage::scale_parameters_dev(MyDevice & dev, float a) {
  values.tvec().device(*dev.edevice) = values.tvec() * a;
}
#ifdef __CUDACC__
template void ParameterStorage::scale_parameters_dev<Device_GPU>(Device_GPU & dev, float a);
#elif defined(HAVE_CUDA)
extern template void ParameterStorage::scale_parameters_dev<Device_GPU>(Device_GPU & dev, float a);
template void ParameterStorage::scale_parameters_dev<Device_CPU>(Device_CPU & dev, float a);
void ParameterStorage::scale_parameters(float a) {
  if (values.device->type == DeviceType::CPU) { scale_parameters_dev(*(Device_CPU*)values.device, a); }
  else if (values.device->type == DeviceType::GPU) { scale_parameters_dev(*(Device_GPU*)values.device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void ParameterStorage::scale_parameters_dev<Device_CPU>(Device_CPU & dev, float a);
void ParameterStorage::scale_parameters(float a) {
  if (values.device->type == DeviceType::CPU) { scale_parameters_dev(*(Device_CPU*)values.device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void ParameterStorage::scale_gradient_dev(MyDevice & dev, float a) {
  g.tvec().device(*dev.edevice) = g.tvec() * a;
}
#ifdef __CUDACC__
template void ParameterStorage::scale_gradient_dev<Device_GPU>(Device_GPU & dev, float a);
#elif defined(HAVE_CUDA)
extern template void ParameterStorage::scale_gradient_dev<Device_GPU>(Device_GPU & dev, float a);
template void ParameterStorage::scale_gradient_dev<Device_CPU>(Device_CPU & dev, float a);
void ParameterStorage::scale_gradient(float a) {
  if (g.device->type == DeviceType::CPU) { scale_gradient_dev(*(Device_CPU*)g.device, a); }
  else if (g.device->type == DeviceType::GPU) { scale_gradient_dev(*(Device_GPU*)g.device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void ParameterStorage::scale_gradient_dev<Device_CPU>(Device_CPU & dev, float a);
void ParameterStorage::scale_gradient(float a) {
  if (g.device->type == DeviceType::CPU) { scale_gradient_dev(*(Device_CPU*)g.device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::initialize_dev(MyDevice & dev, unsigned index, const vector<float>& val) {
  DYNET_ARG_CHECK(int(val.size()) == int(dim.size()),
                          "Attempt to initialize LookupParameters with vector of wrong size "
                          "(" << val.size() << " != " << dim.size() << ")");
#ifdef __CUDACC__
  cudaMemcpyAsync(values[index].v, &val[0], val.size() * sizeof(float), cudaMemcpyHostToDevice);
#else
  memcpy(values[index].v, &val[0], val.size() * sizeof(float));
#endif
}
#ifdef __CUDACC__
template void LookupParameterStorage::initialize_dev<Device_GPU>(Device_GPU & dev, unsigned index, const vector<float>& val);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::initialize_dev<Device_GPU>(Device_GPU & dev, unsigned index, const vector<float>& val);
template void LookupParameterStorage::initialize_dev<Device_CPU>(Device_CPU & dev, unsigned index, const vector<float>& val);
void LookupParameterStorage::initialize(unsigned index, const vector<float>& val) {
  if (values[index].device->type == DeviceType::CPU) { initialize_dev(*(Device_CPU*)values[index].device, index, val); }
  else if (values[index].device->type == DeviceType::GPU) { initialize_dev(*(Device_GPU*)values[index].device, index, val); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::initialize_dev<Device_CPU>(Device_CPU & dev, unsigned index, const vector<float>& val);
void LookupParameterStorage::initialize(unsigned index, const vector<float>& val) {
  if (values[index].device->type == DeviceType::CPU) { initialize_dev(*(Device_CPU*)values[index].device, index, val); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::squared_l2norm_dev(MyDevice & dev, float* sqnorm) const {
  Tensor sqnorm_t({1}, sqnorm, &dev, DeviceMempool::NONE);
  sqnorm_t.t<0>().device(*dev.edevice) = all_values.tvec().square().sum();
}
DYNET_PARAMNORM_INST_DEV_IMPL(LookupParameterStorage, squared_l2norm, squared_l2norm_dev)

template <class MyDevice>
void LookupParameterStorage::g_squared_l2norm_dev(MyDevice & dev, float* sqnorm) const {
  Tensor sqnorm_t({1}, sqnorm, &dev, DeviceMempool::NONE);
  TensorTools::zero(sqnorm_t);
  // TODO: the GPU part is hacky, probably need a better heuristic
  if (all_grads.device->type == DeviceType::GPU || all_updated) {
    sqnorm_t.t<0>().device(*dev.edevice) += all_grads.tvec().square().sum();
  } else {
    auto it = non_zero_grads.begin();
    while (it != non_zero_grads.end())
      sqnorm_t.t<0>().device(*dev.edevice) += grads[*(it++)].tvec().square().sum();
  }
}
DYNET_PARAMNORM_INST_DEV_IMPL(LookupParameterStorage, g_squared_l2norm, g_squared_l2norm_dev)

template <class MyDevice>
void LookupParameterStorage::accumulate_grad_dev(MyDevice & dev, const Tensor& d) {
  all_updated = true;
  all_grads.tvec().device(*dev.edevice) += d.tvec();
}
#ifdef __CUDACC__
template void LookupParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, const Tensor& d);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, const Tensor& d);
template void LookupParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, const Tensor& d);
void LookupParameterStorage::accumulate_grad(const Tensor& d) {
  if (all_values.device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)all_values.device, d); }
  else if (all_values.device->type == DeviceType::GPU) { accumulate_grad_dev(*(Device_GPU*)all_values.device, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, const Tensor& d);
void LookupParameterStorage::accumulate_grad(const Tensor& d) {
  if (all_values.device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)all_values.device, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::accumulate_grad_dev(MyDevice & dev, unsigned index, const Tensor& d) {
  non_zero_grads.insert(index);
  grads[index].tvec().device(*dev.edevice) += d.tvec();
}
#ifdef __CUDACC__
template void LookupParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, unsigned index, const Tensor& d);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::accumulate_grad_dev<Device_GPU>(Device_GPU & dev, unsigned index, const Tensor& d);
template void LookupParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, unsigned index, const Tensor& d);
void LookupParameterStorage::accumulate_grad(unsigned index, const Tensor& d) {
  if (values[index].device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)values[index].device, index, d); }
  else if (values[index].device->type == DeviceType::GPU) { accumulate_grad_dev(*(Device_GPU*)values[index].device, index, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::accumulate_grad_dev<Device_CPU>(Device_CPU & dev, unsigned index, const Tensor& d);
void LookupParameterStorage::accumulate_grad(unsigned index, const Tensor& d) {
  if (values[index].device->type == DeviceType::CPU) { accumulate_grad_dev(*(Device_CPU*)values[index].device, index, d); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::accumulate_grads_dev(MyDevice & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g) {
#ifdef __CUDACC__
  for (unsigned i = 0; i < n; ++i)
    non_zero_grads.insert(ids_host[i]);
  dynet::gpu::dense_to_sparse_block_add(n, ids_dev, dim.size(), g, all_grads.v);
#else
  size_t gsize = dim.size();
  Tensor gt(dim, g, all_grads.device, all_grads.mem_pool);
  for (unsigned i = 0; i < n; ++i) {
    non_zero_grads.insert(ids_host[i]);
    grads[ids_host[i]].tvec().device(*dev.edevice) += gt.tvec();
    gt.v += gsize;
  }
#endif
}
#ifdef __CUDACC__
template void LookupParameterStorage::accumulate_grads_dev<Device_GPU>(Device_GPU & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::accumulate_grads_dev<Device_GPU>(Device_GPU & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
template void LookupParameterStorage::accumulate_grads_dev<Device_CPU>(Device_CPU & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
void LookupParameterStorage::accumulate_grads(unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g) {
  if (all_values.device->type == DeviceType::CPU) { accumulate_grads_dev(*(Device_CPU*)all_values.device, n, ids_host, ids_dev, g); }
  else if (all_values.device->type == DeviceType::GPU) { accumulate_grads_dev(*(Device_GPU*)all_values.device, n, ids_host, ids_dev, g); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::accumulate_grads_dev<Device_CPU>(Device_CPU & dev, unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g);
void LookupParameterStorage::accumulate_grads(unsigned n, const unsigned* ids_host, const unsigned* ids_dev, float* g) {
  if (all_values.device->type == DeviceType::CPU) { accumulate_grads_dev(*(Device_CPU*)all_values.device, n, ids_host, ids_dev, g); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::scale_parameters_dev(MyDevice & dev, float a) {
  all_values.tvec().device(*dev.edevice) = all_values.tvec() * a;
}
#ifdef __CUDACC__
template void LookupParameterStorage::scale_parameters_dev<Device_GPU>(Device_GPU & dev, float a);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::scale_parameters_dev<Device_GPU>(Device_GPU & dev, float a);
template void LookupParameterStorage::scale_parameters_dev<Device_CPU>(Device_CPU & dev, float a);
void LookupParameterStorage::scale_parameters(float a) {
  if (values[0].device->type == DeviceType::CPU) { scale_parameters_dev(*(Device_CPU*)values[0].device, a); }
  else if (values[0].device->type == DeviceType::GPU) { scale_parameters_dev(*(Device_GPU*)values[0].device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::scale_parameters_dev<Device_CPU>(Device_CPU & dev, float a);
void LookupParameterStorage::scale_parameters(float a) {
  if (values[0].device->type == DeviceType::CPU) { scale_parameters_dev(*(Device_CPU*)values[0].device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
void LookupParameterStorage::scale_gradient_dev(MyDevice & dev, float a) {
  all_grads.tvec().device(*dev.edevice) = all_grads.tvec() * a;
}
#ifdef __CUDACC__
template void LookupParameterStorage::scale_gradient_dev<Device_GPU>(Device_GPU & dev, float a);
#elif defined(HAVE_CUDA)
extern template void LookupParameterStorage::scale_gradient_dev<Device_GPU>(Device_GPU & dev, float a);
template void LookupParameterStorage::scale_gradient_dev<Device_CPU>(Device_CPU & dev, float a);
void LookupParameterStorage::scale_gradient(float a) {
  if (grads[0].device->type == DeviceType::CPU) { scale_gradient_dev(*(Device_CPU*)grads[0].device, a); }
  else if (grads[0].device->type == DeviceType::GPU) { scale_gradient_dev(*(Device_GPU*)grads[0].device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template void LookupParameterStorage::scale_gradient_dev<Device_CPU>(Device_CPU & dev, float a);
void LookupParameterStorage::scale_gradient(float a) {
  if (grads[0].device->type == DeviceType::CPU) { scale_gradient_dev(*(Device_CPU*)grads[0].device, a); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

template <class MyDevice>
float Model::gradient_l2_norm_dev(MyDevice & dev) const {
  if (!gradient_norm_scratch)
    gradient_norm_scratch = (float*)default_device->mem->malloc((all_params.size() + 1) * sizeof(float));
  size_t pi;
  for (pi = 0; pi < all_params.size(); ++pi)
    all_params[pi]->g_squared_l2norm(&gradient_norm_scratch[pi]);
  Tensor scratch_t({(unsigned int)all_params.size()}, gradient_norm_scratch, &dev, DeviceMempool::NONE);
  Tensor sum_t({1}, gradient_norm_scratch + pi, &dev, DeviceMempool::NONE);
  sum_t.t<0>().device(*dev.edevice) = scratch_t.t<1>().sum().sqrt();
#ifdef __CUDACC__
  float res = 0;
  cudaMemcpy(&res, gradient_norm_scratch + pi, sizeof(float),  cudaMemcpyDeviceToHost);
  return res;
#else
  return gradient_norm_scratch[pi];
#endif
}
#ifdef __CUDACC__
template float Model::gradient_l2_norm_dev<Device_GPU>(Device_GPU & dev) const;
#elif defined(HAVE_CUDA)
extern template float Model::gradient_l2_norm_dev<Device_GPU>(Device_GPU & dev) const;
template float Model::gradient_l2_norm_dev<Device_CPU>(Device_CPU & dev) const;
float Model::gradient_l2_norm() const {
  if (default_device->type == DeviceType::CPU) { return gradient_l2_norm_dev(*(Device_CPU*)default_device); }
  else if (default_device->type == DeviceType::GPU) { return gradient_l2_norm_dev(*(Device_GPU*)default_device); }
  else { throw std::runtime_error("Bad device type"); }
}
#else
template float Model::gradient_l2_norm_dev<Device_CPU>(Device_CPU & dev) const;
float Model::gradient_l2_norm() const {
  if (default_device->type == DeviceType::CPU) { return gradient_l2_norm_dev(*(Device_CPU*)default_device); }
  else { throw std::runtime_error("Bad device type"); }
}
#endif

} // namespace dynet

#ifndef __CUDACC__
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::ParameterStorage)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::LookupParameterStorage)
#endif
