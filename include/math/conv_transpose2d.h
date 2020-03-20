#ifndef MATH_CONV_TRANSPOSE2D_H
#define MATH_CONV_TRANSPOSE2D_H

#include "common/tensor.h"

namespace dlfm::math {

// input [n, c, h, w]
// weight [in_c, out_c, kernel_size[0], kernel_size[1]]
// bias [out_c]
// output [n,
//         out_c,
//         (h - 1) * stride[0] - 2 * padding[0] + kernel[0] + output_padding[0]
//         (h - 1) * stride[1] - 2 * padding[1] + kernel[1] + output_padding[1]]
void conv_transpose2d(const Tensor &input,
                      const Tensor &weight,
                      const Tensor &bias,
                      Tensor &output,
                      std::vector<size_t> kernel_size,
                      std::vector<size_t> stride,
                      std::vector<size_t> padding,
                      std::vector<size_t> out_padding);

}

#endif //DLFM_CONV_TRANSPOSE2D_H
