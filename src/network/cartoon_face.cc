#include "common/tensor.h"
#include "module/conv2d.h"
#include "module/max_pooling2d.h"
#include "module/relu.h"
#include "module/tanh.h"
#include "module/sequential.h"
#include "module/sigmoid.h"
#include "module/upsample2d.h"
#include "module/instance_norm2d.h"
#include "network/cartoon_face.h"

namespace dlfm::nn::cartoon_face {

ConvBlock::ConvBlock(int64_t in, int64_t out) {
  dim_out = out;

  ADD_SUB_MODULE(ConvBlock1, sequential, {
    instance_norm2d(in, 1e-05, false),
    relu(true),
    reflection_pad2d(1),
    conv2d(in, out / 2, 3, 1, 0)
  });

  ADD_SUB_MODULE(ConvBlock2, sequential, {
    instance_norm2d(out / 2, 1e-05, false),
    relu(true),
    reflection_pad2d(1),
    conv2d(out / 2, out / 4, 3, 1, 0)
  });

  ADD_SUB_MODULE(ConvBlock3, sequential, {
    instance_norm2d(out / 4, 1e-05, false),
    relu(true),
    reflection_pad2d(1),
    conv2d(out / 4, out / 4, 3, 1, 0)
  });

  ADD_SUB_MODULE(ConvBlock4, sequential, {
    instance_norm2d(in, 1e-05, false),
    relu(true),
    conv2d(in, out, 1, 1, 0)
  });
}

Tensor ConvBlock::forward(Tensor x) {
  auto residual = x;

  auto x1 = (*ConvBlock1)(x);
  auto x2 = (*ConvBlock2)(x1);
  auto x3 = (*ConvBlock3)(x2);

  auto out = x1.cat(x2, 1).cat(x3, 1);

  if (residual.shape()[1] != dim_out) {
    residual = (*ConvBlock4)(residual);
  }

  return residual + out;
}

}