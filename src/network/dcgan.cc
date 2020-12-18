#include "network/dcgan.h"

namespace dlfm::nn::dcgan {

ResidualBlock::ResidualBlock(
  int64_t in_channels,
  int64_t out_channels,
  int64_t kernel_size,
  int64_t stride,
  int64_t padding,
  bool bias) {
  ADD_SUB_MODULE(conv_1, conv2d, in_channels, out_channels, kernel_size, stride, padding, 1, 1, false);
  ADD_SUB_MODULE(bn_1, batch_norm2d, out_channels, 1e-05, true, true);
  ADD_SUB_MODULE(relu, prelu, true);
  ADD_SUB_MODULE(conv_1, conv2d, out_channels, out_channels, kernel_size, stride, padding, 1, 1, false);
  ADD_SUB_MODULE(bn_1, batch_norm2d, out_channels, 1e-05, true, true);
}

Tensor ResidualBlock::forward(Tensor x) {
  auto r_tensor = x;
  auto output = (*conv_1)(x);
  output = (*bn_1)(output);
  output = (*relu)(output);
  output = (*conv_2)(output);
  output = (*bn_2)(output);
  output += r_tensor;

  return output;
}

SubpixelBlock::SubpixelBlock(
  int64_t in_channels,
  int64_t out_channels,
  int64_t kernel_size,
  int64_t stride,
  int64_t padding,
  bool bias,
  int64_t up): upscale_factor(up) {
  ADD_SUB_MODULE(conv, conv2d, in_channels, out_channels, kernel_size, stride, padding, 1, 1, bias);
  ADD_SUB_MODULE(bn, batch_norm2d, in_channels, 1e-05, true, true);
  ADD_SUB_MODULE(relu, prelu, true);
}

Tensor SubpixelBlock::forward(Tensor x) {
  auto output = (*conv)(x);
  output = output.pixel_shuffle(upscale_factor);
  output = (*bn)(output);
  output = (*relu)(output);

  return output;
}

Generator::Generator(int64_t tag, int64_t residual_block_size, int64_t subpixel_block_size) {
  in_channels = 128 + tag;

  ADD_SUB_MODULE(dense_1, linear, in_channels, 64 * 16 * 16);
  ADD_SUB_MODULE(bn_1, batch_norm2d, 64, 1e-05, true, true);
  ADD_SUB_MODULE(relu_1, prelu, true);

  std::vector<Module> residual_layer_items;
  for (int i = 0; i < residual_block_size; ++i) {
    residual_layer_items.emplace_back(std::make_shared<ResidualBlock>(64, 64, 3, 1));
  }

  ADD_SUB_MODULE(residual_layer, sequential, residual_layer_items);

  ADD_SUB_MODULE(bn_2, batch_norm2d, 64, 1e-05, true, true);
  ADD_SUB_MODULE(relu_2, prelu, true);

  std::vector<Module> subpixel_layer_items;
  for (int i = 0; i < subpixel_block_size; ++i) {
    subpixel_layer_items.emplace_back(std::make_shared<SubpixelBlock>(64, 256, 3, 1));
  }

  ADD_SUB_MODULE(subpixel_layer, sequential, subpixel_layer_items);

  ADD_SUB_MODULE(conv_1, conv2d, 64, 3, 9, 1, 4, 1, 1, true);
  ADD_SUB_MODULE(tanh_1, tanh, true);
}

Tensor Generator::forward(Tensor x) {
  auto output = (*dense_1)(x);

  output = output.view({-1, 64, 16, 16});
  output = (*bn_1)(output);
  output = (*relu_1)(output);

  auto r_output = output;

  output = (*residual_layer)(output);
  output = (*bn_2)(output);
  output = (*relu_2)(output);

  output += r_output;

  output = (*subpixel_layer)(output);
  output = (*conv_1)(output);
  output = (*tanh_1)(output);

  return output;
}

}