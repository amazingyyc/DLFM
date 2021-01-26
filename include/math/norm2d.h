#ifndef MATH_NORM2D_H
#define MATH_NORM2D_H

#include "common/tensor.h"

namespace dlfm::math {

// A common norm for 2d tensor.
// x shape must be 2 dimension.
void norm2d(const Tensor &x, Tensor &y, float eps);

}

#endif