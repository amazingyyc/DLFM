#ifndef MATH_RELU_H
#define MATH_RELU_H

#include "common/tensor.h"

namespace dlfm::math {

void relu(const Tensor &x, Tensor &y);

}

#endif