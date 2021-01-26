#ifndef MATH_UNARY_CWISE_H
#define MATH_UNARY_CWISE_H

#include "common/tensor.h"

namespace dlfm::math {

void assign(const Tensor&, Tensor&);

void add(const Tensor&, float, Tensor&);

void sub(const Tensor&, float, Tensor&);

void multiply(const Tensor&, float, Tensor&);

void divide(const Tensor&, float, Tensor&);

void add(float, const Tensor&, Tensor&);

void sub(float, const Tensor&, Tensor&);

void multiply(float, const Tensor&, Tensor&);

void divide(float, const Tensor&, Tensor&);

// y = x // val
void floor_divide(const Tensor &x, Tensor &y, float val);

// y = x % x
void remainder(const Tensor &x, Tensor &y, float val);

}

#endif