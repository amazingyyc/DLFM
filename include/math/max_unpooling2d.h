#ifndef MATH_MAX_UNPOOLING2D_H
#define MATH_MAX_UNPOOLING2D_H

#include "common/tensor.h"

namespace dlfm::math {

void max_unpooling2d(const Tensor &x, const Tensor &indices, Tensor &y);

}

#endif