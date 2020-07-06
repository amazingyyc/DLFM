#ifndef VISION_ROTATE_H
#define VISION_ROTATE_H

#include "common/basic.h"
#include "common/element_type.h"
#include "common/tensor.h"

namespace dlfm::vision {

// rotate the image right_90 degree
Tensor rotate_right_90(const Tensor &x);

}

#endif