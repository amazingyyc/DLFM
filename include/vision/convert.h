#ifndef VISION_CONVERT_H
#define VISION_CONVERT_H

#include "common/basic.h"
#include "common/element_type.h"
#include "common/tensor.h"

namespace dlfm::vision {

// convert channel like RGB->BGR.
Tensor convert(const Tensor &x, std::vector<size_t> idx);

Tensor bgra_2_rgb(const Tensor &x);

Tensor bgra_2_rgba(const Tensor &x);

Tensor rgbx_2_rgb(const Tensor &x);

// yuv to rgb full range.
// use float to calculate
Tensor yuv_2_rgb_full(const Tensor &y, const Tensor &uv);

// yuv to rgb video range.
// use float to calculate
Tensor yuv_2_rgb_video(const Tensor &y, const Tensor &uv);

}

#endif