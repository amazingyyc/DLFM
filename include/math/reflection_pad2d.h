#ifndef MATH_REFLECTIONPAD2D_H
#define MATH_REFLECTIONPAD2D_H

#include "common/tensor.h"

namespace dlfm::math {

void reflection_pad2d(const Tensor&, Tensor&, std::vector<size_t>);

}

#endif

