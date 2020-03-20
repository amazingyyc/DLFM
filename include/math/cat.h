#ifndef MATH_CAT_H
#define MATH_CAT_H

#include "common/tensor.h"

namespace dlfm::math {

void cat(const Tensor&, const Tensor&, Tensor&, int64_t);

}

#endif