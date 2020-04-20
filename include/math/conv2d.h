#ifndef MATH_CONV2D_H
#define MATH_CONV2D_H

#include "common/tensor.h"

namespace dlfm::math {

void conv2d(const Tensor &input, const Tensor &weight, const Tensor &bias, Tensor &output, std::vector<size_t> stride, std::vector<size_t> padding, int64_t groups);

}
#endif