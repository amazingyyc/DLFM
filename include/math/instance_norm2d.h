#ifndef MATH_INSTANCE_NORM_H
#define MATH_INSTANCE_NORM_H

#include "common/tensor.h"

namespace dlfm::math {

// instance norm2d need input is [b, c, h, w],
void instance_norm2d(const Tensor &input, float eps, Tensor &output);

// instance norm2d need input is [b, c, h, w], outout[b, c, h, w], scale[c], shift[c]
void instance_norm2d(const Tensor &input, const Tensor &scale, const Tensor &shift, float eps, Tensor &output);

}

#endif