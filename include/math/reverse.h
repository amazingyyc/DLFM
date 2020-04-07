#ifndef MATH_REVERSE_H
#define MATH_REVERSE_H

#include "common/tensor.h"

namespace dlfm::math {

void reverse(const Tensor &x, Tensor &y, const std::vector<bool>&);

}

#endif