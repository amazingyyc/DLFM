#ifndef MATH_ADAPTIVE_MAX_POOLING2D_H
#define MATH_ADAPTIVE_MAX_POOLING2D_H

#include "common/tensor.h"

namespace dlfm::math {

void adaptive_max_pooling2d(const Tensor&, Tensor&);

}

#endif