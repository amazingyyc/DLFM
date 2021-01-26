#include "network/pfld_lite.h"

namespace dlfm::nn::pfld_lite {

Sequential conv_bn(
  int64_t inp,
  int64_t oup,
  int64_t stride) {
  return sequential({
    conv2d(inp, oup, 3, stride, 1, 1, 1, false),
    batch_norm2d(oup),
    relu(true)
  });
}

Sequential conv_bn1(
  int64_t inp,
  int64_t oup,
  int64_t kernel,
  int64_t stride,
  int64_t padding) {
  return sequential({
    conv2d(inp, oup, kernel, stride, padding, 1, 1, false),
    batch_norm2d(oup),
    relu(true)
  });
}

Sequential conv_dw(
  int64_t inp,
  int64_t oup,
  int64_t stride) {
  return sequential({
    conv2d(inp, inp, 3, stride, 1, 1, inp, false),
    batch_norm2d(inp),
    relu(true),

    conv2d(inp, oup, 1, 1, 0, 1, 1, false),
    batch_norm2d(oup),
    relu(true),
  });
}

Sequential SeperableConv2d(
  int64_t in_channels,
  int64_t out_channels,
  int64_t kernel_size,
  int64_t stride,
  int64_t padding) {
  return sequential({
    conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, in_channels, true),
    relu(true),

    conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, true)
  });
}

PFLDInference::PFLDInference() {
  base_channel = 16;

  ADD_SUB_MODULE(model, sequential, {
    conv_bn(3, base_channel, 1),  // 56 * 56
    conv_dw(base_channel, base_channel * 2, 1),
    conv_dw(base_channel * 2, base_channel * 2, 2),
    conv_dw(base_channel * 2, base_channel * 2, 1),
    conv_dw(base_channel * 2, base_channel * 4, 2),
    conv_dw(base_channel * 4, base_channel * 4, 1),
    conv_dw(base_channel * 4, base_channel * 4, 1),
    conv_dw(base_channel * 4, base_channel * 4, 1),
  });

  ADD_SUB_MODULE(model1, sequential, {
    conv_dw(base_channel * 4, base_channel * 8, 2),
    conv_dw(base_channel * 8, base_channel * 8, 1),
    conv_dw(base_channel * 8, base_channel * 8, 1),
    conv_dw(base_channel * 8, base_channel * 16, 1),
    conv_dw(base_channel * 16, base_channel * 16, 1)
  });

  ADD_SUB_MODULE(extra, sequential, {
    conv2d(base_channel * 16, base_channel * 4, 1, 1, 0, 1, 1, true),
    relu(true),
    SeperableConv2d(base_channel * 4, base_channel * 8, 3, 2, 1)
  });

  ADD_SUB_MODULE(conv1, conv2d, 64, 16, 3, 1, 1, 1, 1, true);
  ADD_SUB_MODULE(conv2, conv2d, 256, 64, 1, 1, 0, 1, 1, true);

  ADD_SUB_MODULE(fc, linear, 208, 106 * 2);
}

// x: [1, 3, 112, 112]
Tensor PFLDInference::forward(Tensor x) {
  auto out = (*model)(x);

  auto x1 = (*conv1)(out);
  x1 = x1.adaptive_avg_pooling2d(1);
  x1 = x1.view({x1.shape()[0], -1});

  x = (*model1)(out);
  auto x2 = (*conv2)(x);
  x2 = x2.adaptive_avg_pooling2d(1);
  x2 = x2.view({ x2.shape()[0], -1});

  x = (*extra)(x);
  x = x.adaptive_avg_pooling2d(1);
  auto x3 = x.view({ x.shape()[0], -1 });

  auto multi_scale = x1.cat({ x2, x3 }, 1);

  auto landmarks = (*fc)(multi_scale);

  return landmarks;
}

}