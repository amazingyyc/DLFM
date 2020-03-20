#ifndef MATH_MAX_POOLING2D_H
#define MATH_MAX_POOLING2D_H

#include "common/tensor.h"

namespace dlfm::math {

void max_pooling2d(const Tensor&, Tensor&, std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding);

}

#endif