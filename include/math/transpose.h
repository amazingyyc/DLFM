#ifndef MATH_TRANSPOSE_H
#define MATH_TRANSPOSE_H

#include "common/tensor.h"

namespace dlfm::math {

void transpose(const Tensor&, Tensor&, std::vector<size_t>);

}

#endif