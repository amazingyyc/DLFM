#ifndef VISION_PAD_H
#define VISION_PAD_H

#include "common/basic.h"
#include "common/element_type.h"
#include "common/tensor.h"

namespace dlfm::vision {

// pad image
// paddings: [top, bottom, left, right]
Tensor pad(const Tensor &x, const Tensor &value, int64_t top, int64_t bottom, int64_t left, int64_t right);

Tensor pad(const Tensor &x, const Tensor &value, std::vector<int64_t> paddings);

}

#endif