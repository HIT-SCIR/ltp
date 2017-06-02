#ifndef DYNET_EIGEN_INIT_H
#define DYNET_EIGEN_INIT_H

#include <string>
#include <vector>

namespace dynet {

extern float weight_decay_lambda;

/**
 * \brief Represents general parameters for dynet
 *
 */
struct DynetParams {
  DynetParams();
  ~DynetParams();
  unsigned random_seed = 0; /**< The seed for random number generation */
  std::string mem_descriptor = "512"; /**< Total memory to be allocated for Dynet */
  float weight_decay = 0; /**< Weight decay rate for L2 regularization */
  bool shared_parameters = false; /**< TO DOCUMENT */
  bool ngpus_requested = false; /**< GPUs requested by number */
  bool ids_requested = false; /**< GPUs requested by ids */
  int requested_gpus = -1; /**< Number of requested GPUs */
  std::vector<int> gpu_mask; /**< List of required GPUs by ids */


};

DynetParams extract_dynet_params(int& argc, char**& argv, bool shared_parameters = false);
void initialize(DynetParams& params);
void initialize(int& argc, char**& argv, bool shared_parameters = false);
void cleanup();

} // namespace dynet

#endif
