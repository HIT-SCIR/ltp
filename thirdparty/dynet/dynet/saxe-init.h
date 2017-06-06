#ifndef DYNET_SAXE_INIT_H_
#define DYNET_SAXE_INIT_H_

namespace dynet {

struct Tensor;

void orthonormal_random(unsigned dim, float g, Tensor& x);

}

#endif
