#ifndef VISION_RESAMPLE_H
#define VISION_RESAMPLE_H

#include "common/basic.h"
#include "common/element_type.h"
#include "common/tensor.h"

namespace dlfm::vision {

// vision is common API for Computer Vision, especially for Image process.
// All input tensor shape must be [h, w, pixel format]

// resize a image, img must be float or uint8
// mode: nearest/n, bilinear/b
Tensor resize(const Tensor &img, std::vector<int64_t> size, std::string mode = "b", bool align_corners = false);

// same resize with mode is "n"
Tensor resample_nearest(const Tensor &img, std::vector<int64_t> size);

// same resize with mode is "b"
Tensor resample_bilinear(const Tensor &img, std::vector<int64_t> size, bool align_corners = false);

}

#endif