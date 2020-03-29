#ifndef MATH_UPSAMPLE2D_H
#define MATH_UPSAMPLE2D_H

#include "common/tensor.h"

namespace dlfm::math {

// nearest only support align_corners is false
void upsample_nearest2d(const Tensor &, Tensor &);

void upsample_bilinear2d(const Tensor &, Tensor &, bool align_corners);

}

#endif