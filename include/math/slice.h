#ifndef MATH_SLICE_H
#define MATH_SLICE_H

#include "common/tensor.h"

namespace dlfm::math {

void slice(const Tensor &x, Tensor &y, std::vector<int64_t> offsets, std::vector<int64_t> extents);

}

#endif