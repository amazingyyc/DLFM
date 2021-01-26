#ifndef MATH_TOPK_H
#define MATH_TOPK_H

#include "common/tensor.h"

namespace dlfm::math {

void topk(const Tensor &x, Tensor &y, Tensor &indices, int64_t k, int64_t axis, bool largest, bool sorted);

}

#endif