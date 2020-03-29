#ifndef MATH_UNARY_CWISE_H
#define MATH_UNARY_CWISE_H

#include "common/tensor.h"

namespace dlfm::math {

void assign(const Tensor&, Tensor&);

void add(const Tensor&, float, Tensor&);

void sub(const Tensor&, float, Tensor&);

void multiply(const Tensor&, float, Tensor&);

void divide(const Tensor&, float, Tensor&);

}

#endif