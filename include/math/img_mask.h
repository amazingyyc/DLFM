#ifndef MATH_IMG_MASK_H
#define MATH_IMG_MASK_H

#include "common/tensor.h"

namespace dlfm::math {

// add mask to x, if the mask it 1 than retain the x or set to val
// x shape is [c, h, w]
// mask is [h, w] must be uint8_t element type
// val shape must be [c]
// y shape is [c, h, w]
void img_mask(const Tensor &x, const Tensor &mask, const Tensor &val, Tensor &y);

}

#endif