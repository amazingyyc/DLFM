#ifndef MATH_BATCH_NORM2D_H
#define MATH_BATCH_NORM2D_H

#include "common/tensor.h"

namespace dlfm::math {

// instance norm2d need input is [b, c, h, w], outout[b, c, h, w]
void batch_norm2d(
  const Tensor &input, // [b, c, h, w]
  const Tensor &mean, // [1, c, 1, 1]
  const Tensor &variance, // [1, c, 1, 1]
  const Tensor &scale, // [1, c, 1, 1]
  const Tensor &shift, // [1, c, 1 ,1]
  float eps,
  Tensor &output);

}

#endif