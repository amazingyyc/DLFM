#ifndef MATH_PAD_H
#define MATH_PAD_H

#include "common/tensor.h"

namespace dlfm::math {

void pad(const Tensor&, Tensor&, std::vector<size_t>);

}

#endif

