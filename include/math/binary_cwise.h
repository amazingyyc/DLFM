#ifndef MATH_BINARY_CWISE_H
#define MATH_BINARY_CWISE_H

#include "common/tensor.h"

namespace dlfm::math {

void add(const Tensor&, const Tensor&, Tensor&);

void sub(const Tensor&, const Tensor&, Tensor&);

void multiply(const Tensor&, const Tensor&, Tensor&);

void divide(const Tensor&, const Tensor&, Tensor&);

}

#endif