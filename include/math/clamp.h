#ifndef MATH_CLAMP_H
#define MATH_CLAMP_H

#include "common/tensor.h"

namespace dlfm::math {

void clamp(const Tensor&, Tensor&, float min, float max);

}

#endif