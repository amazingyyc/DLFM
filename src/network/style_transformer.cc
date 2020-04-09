#include "module/reflection_pad2d.h"

#include "network/style_transformer.h"

namespace dlfm::nn::style_transformer {

ConvBlock::ConvBlock(
  int64_t in_channels,
  int64_t out_channels,
  int64_t kernel_size,
  int64_t stride,
  bool will_upsample,
  bool will_normalize,
  bool will_relu) {
  upsample = will_upsample;
  relu = will_relu;

  ADD_SUB_MODULE(block, sequential, {
    reflection_pad2d(kernel_size / 2),
    conv2d(in_channels, out_channels, kernel_size, stride)
  });

  if (will_normalize) {
    ADD_SUB_MODULE(norm, instance_norm2d, out_channels, 1e-05, true);
  }
}

Tensor ConvBlock::forward(Tensor x) {
  // torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
  if (upsample) {
    x = x.upsample2d(2.0);
  }

  x = (*block)(x);

  if (norm) {
    x = (*norm)(x);
  }

  if (relu) {
    x.relu(true);
  }

  return x;
}

ResidualBlock::ResidualBlock(int64_t channels) {
  ADD_SUB_MODULE(block, sequential, {
    std::make_shared<ConvBlock>(channels, channels, 3, 1, false, true, true),
    std::make_shared<ConvBlock>(channels, channels, 3, 1, false, true, true)
  });
}

Tensor ResidualBlock::forward(Tensor x) {
  return (*block)(x) + x;
}

Transformer::Transformer() {
  ADD_SUB_MODULE(model, sequential, {
    std::make_shared<ConvBlock>(3, 32, 9, 1),
    std::make_shared<ConvBlock>(32, 64, 3, 2),
    std::make_shared<ConvBlock>(64, 128, 3, 2),
    std::make_shared<ResidualBlock>(128),
    std::make_shared<ResidualBlock>(128),
    std::make_shared<ResidualBlock>(128),
    std::make_shared<ResidualBlock>(128),
    std::make_shared<ResidualBlock>(128),
    std::make_shared<ConvBlock>(128, 64, 3, 1, true),
    std::make_shared<ConvBlock>(64, 32, 3, 1, true),
    std::make_shared<ConvBlock>(32, 3, 9, 1, false, false, false),
  });
}

Tensor Transformer::forward(Tensor x) {
  return (*model)(x);
}

}