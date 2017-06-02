#ifndef DYNET_GLOBALS_H
#define DYNET_GLOBALS_H

#include <random>
#include <vector>

namespace dynet {

class Device;

extern std::mt19937* rndeng;
extern std::vector<Device*> devices;
extern Device* default_device;

} // namespace dynet

#endif
