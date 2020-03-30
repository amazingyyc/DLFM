#ifndef MATH_MATMUL_H
#define MATH_MATMUL_H

#include "common/tensor.h"

namespace dlfm::math {

void matmul(const Tensor&, const Tensor&, Tensor&, bool transpose_a, bool transpose_b);

}

#endif