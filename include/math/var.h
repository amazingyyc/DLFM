#ifndef MATH_VAR_H
#define MATH_VAR_H

#include "common/tensor.h"

namespace dlfm::math {

void var(const Tensor &x, const Tensor &mean, int64_t axis, bool unbiased, Tensor &y);

}

#endif
