#ifndef MATH_RELU_H
#define MATH_RELU_H

#include "common/tensor.h"

namespace dlfm::math {

void relu(const Tensor &x, Tensor &y);

void relu6(const Tensor &x, Tensor &y);

void prelu(const Tensor &x, const Tensor &w, Tensor &y);

}

#endif