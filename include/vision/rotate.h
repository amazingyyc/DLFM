#ifndef VISION_ROTATE_H
#define VISION_ROTATE_H

#include "common/basic.h"
#include "common/element_type.h"
#include "common/tensor.h"

namespace dlfm::vision {

// rotate the image right_90 degree
Tensor rotate_right_90(const Tensor &x);

// anticlockwise rotate matirx, anlge is radians.
// origin is image center and size will not change.
// [cos(a),   sin(a), -(w/2)cos(a) - (h/2)sin(a) + w/2]
// [-sin(a),  cos(a),  (w/2)sin(a) - (h/2)cos(a) + h/2]
// [     0,       0,                                1]
Tensor rotate(const Tensor &x, float angle, const Tensor &pad);

// Affine transform
// matrix: [m00, m01, m02, m11, m11, m12]
Tensor affine_transform(const Tensor &x, const std::vector<float> &matrix, const Tensor &pad);

}

#endif