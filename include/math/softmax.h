#ifndef MATH_SOFTMAX_H
#define MATH_SOFTMAX_H

#include "common/tensor.h"

namespace dlfm::math {

void softmax(const Tensor &x, Tensor &y, int64_t axis);

}

#endif