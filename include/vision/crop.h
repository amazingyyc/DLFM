#ifndef VISION_CROP_H
#define VISION_CROP_H

#include "common/basic.h"
#include "common/element_type.h"
#include "common/tensor.h"

namespace dlfm::vision {

// coordinate: left_top
// offset: [y, x], size:[height, width]
Tensor crop(const Tensor &x, std::vector<int64_t> offset, std::vector<int64_t> size);

}

#endif