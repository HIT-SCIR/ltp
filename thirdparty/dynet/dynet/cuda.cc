#include <iostream>
#include <vector>
#include <algorithm>

#include "dynet/dynet.h"
#include "dynet/cuda.h"
#include "dynet/init.h"

using namespace std;

namespace dynet {

vector<Device*> initialize_gpu(DynetParams& params) {
  // Get GPU devices count
  int nDevices;
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));
  if (nDevices < 1)
    throw std::runtime_error("No GPUs found but DyNet compiled with CUDA support. Recompile without -DBACKEND=cuda");

  // Check gpu_mask
  for (unsigned gpu_id = nDevices; gpu_id < MAX_GPUS; ++gpu_id) {
    if (params.gpu_mask[gpu_id] != 0) {
      ostringstream oss; oss << "You requested GPU id " << gpu_id << " but system only reports up to " << nDevices;
      throw std::invalid_argument(oss.str());
    }
  }

  if (params.ngpus_requested || params.requested_gpus == -1) {
    if (params.requested_gpus == -1) params.requested_gpus = 1;
    cerr << "Request for " << params.requested_gpus << " GPU" << (params.requested_gpus == 1 ? "" : "s") << " ...\n";
    for (int i = 0; i < MAX_GPUS; ++i) params.gpu_mask[i] = 1;
  } else if (params.ids_requested) {
    params.requested_gpus++;
    cerr << "[dynet] Request for " << params.requested_gpus << " specific GPU" << (params.requested_gpus == 1 ? "" : "s") << " ...\n";
  }

  vector<Device*> gpudevices;
  if (params.requested_gpus == 0) return gpudevices;
  if (params.requested_gpus > nDevices) {
    ostringstream oss; oss << "You requested " << params.requested_gpus << " GPUs but system only reports " << nDevices;
    throw std::invalid_argument(oss.str());
  }

  // after all that, params.requested_gpus is the number of GPUs to reserve
  // we now pick the ones that are both requested by the user or have
  // the most memory free

  vector<size_t> gpu_free_mem(MAX_GPUS, 0);
  vector<int> gpus(MAX_GPUS, 0);
  for (int i = 0; i < MAX_GPUS; ++i) gpus[i] = i;
  size_t free_bytes, total_bytes;
  for (int i = 0; i < nDevices; i++) {
    if (!params.gpu_mask[i]) continue;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    cerr << "[dynet] Device Number: " << i << endl;
    cerr << "[dynet]   Device name: " << prop.name << endl;
    cerr << "[dynet]   Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
    cerr << "[dynet]   Memory Bus Width (bits): " << prop.memoryBusWidth << endl;
    cerr << "[dynet]   Peak Memory Bandwidth (GB/s): " << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6) << endl;
    if (!prop.unifiedAddressing)
      throw std::invalid_argument("[dynet] GPU does not support unified addressing.");
    CUDA_CHECK(cudaSetDevice(i));
    try {
      CUDA_CHECK(cudaMemGetInfo( &free_bytes, &total_bytes ));
      cerr << "[dynet]   Memory Free (GB): " << free_bytes / 1.0e9 << "/" << total_bytes / 1.0e9 << endl;
      cerr << "[dynet]" << endl;
      gpu_free_mem[i] = free_bytes;
    } catch (dynet::cuda_exception e) {
      cerr << "[dynet]   FAILED to get free memory" << endl;
      gpu_free_mem[i] = 0;
      cudaGetLastError();
    }
    CUDA_CHECK(cudaDeviceReset());
  }
  stable_sort(gpus.begin(), gpus.end(), [&](int a, int b) -> bool { return gpu_free_mem[a] > gpu_free_mem[b]; });
  gpus.resize(params.requested_gpus);
  cerr << "[dynet] Device(s) selected:";
  for (int i = 0; i < params.requested_gpus; ++i) {
    cerr << ' ' << gpus[i];
    Device* d = new Device_GPU(gpudevices.size(), params.mem_descriptor, gpus[i]);
    gpudevices.push_back(d);
  }
  cerr << endl;

  return gpudevices;

}

vector<Device*> initialize_gpu(int& argc, char**& argv) {
  DynetParams params = extract_dynet_params(argc, argv);
  return initialize_gpu(params);
}

} // namespace dynet
