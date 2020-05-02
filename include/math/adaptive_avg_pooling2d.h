#ifndef MATH_ADAPTIVE_AVG_POOLING2D_H
#define MATH_ADAPTIVE_AVG_POOLING2D_H

#include "common/tensor.h"

namespace dlfm::math {

void adaptive_avg_pooling2d(const Tensor&, Tensor&);

}

#endif