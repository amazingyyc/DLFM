#include "module/prelu.h"
#include "module/tanh.h"
#include "network/srgan.h"

namespace dlfm::nn::srgan {

ConvolutionalBlock::ConvolutionalBlock(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, bool batch_norm, std::string activation) {
  std::vector<Module> layers;

  layers.emplace_back(conv2d(in_channels, out_channels, kernel_size, stride, kernel_size / 2));

  if (batch_norm) {
    layers.emplace_back(batch_norm2d(out_channels));
  }

  if ("prelu" == activation) {
    layers.emplace_back(prelu(true));
  } else if ("tanh" == activation) {
    layers.emplace_back(tanh(true));
  } else if ("" != activation) {
    RUNTIME_ERROR("activation is error");
  }

  ADD_SUB_MODULE(conv_block, sequential, layers);
}

Tensor ConvolutionalBlock::forward(Tensor x) {
  return (*conv_block)(x);
}

SubPixelConvolutionalBlock::SubPixelConvolutionalBlock(int64_t kernel_size, int64_t n_channels, int64_t scaling) {
  ADD_SUB_MODULE(conv, conv2d, n_channels, n_channels * scaling_factor * scaling_factor, kernel_size, 1, kernel_size / 2);
  ADD_SUB_MODULE(prelu, std::make_shared<PReluImpl>, true);

  scaling_factor = scaling;
}

Tensor SubPixelConvolutionalBlock::forward(Tensor input) {
  auto output = (*conv)(intput);
  output = output.pixel_shuffle(scaling_factor);

  return (*prelu)(output);
}

ResidualBlock::ResidualBlock(int64_t kernel_size, int64_t n_channels) {
  ADD_SUB_MODULE(conv_block1, std::make_shared<ConvolutionalBlock>, n_channels, n_channels, kernel_size, 1, true, "prelu");
  ADD_SUB_MODULE(conv_block2, std::make_shared<ConvolutionalBlock>, n_channels, n_channels, kernel_size, 1, true, "");
}

Tensor ResidualBlock::forward(Tensor input) {
  auto residual = input;

  auto output = (*conv_block1)(input);
  output = (*conv_block2)(output);

  return output + residual;
}

SRResNet::SRResNet(int64_t large_kernel_size, int64_t small_kernel_size, int64_t n_channels, int64_t n_blocks, int64_t scaling_factor) {
  ARGUMENT_CHECK(4 == scaling_factor, "scaling_factor must be 4");

  ADD_SUB_MODULE(conv_block1, std::make_shared<ConvolutionalBlock>, 3, n_channels, large_kernel_size, 1, false, "prelu");

  std::vector<Module> residual_blocks_ms;

  for (int64_t i = 0; i < n_blocks; ++i) {
    residual_blocks_ms.emplace_back(std::make_shared<ResidualBlock>(small_kernel_size, n_channels));
  }

  ADD_SUB_MODULE(residual_blocks, sequential, residual_blocks_ms);

  ADD_SUB_MODULE(conv_block2, std::make_shared<ConvolutionalBlock>, n_channels, n_channels, small_kernel_size, 1, false, "");

  int64_t n_subpixel_convolution_blocks = int64_t(log2f(float(scaling_factor)));
  std::vector<Module> subpixel_convolutional_blocks_ms;
  for (int64_t i = 0; i < n_subpixel_convolution_blocks; ++i) {
    subpixel_convolutional_blocks_ms.emplace_back(std::make_shared<SubPixelConvolutionalBlock>(small_kernel_size, n_channels, 2));
  }

  ADD_SUB_MODULE(subpixel_convolutional_blocks, sequential, subpixel_convolutional_blocks_ms);

  ADD_SUB_MODULE(conv_block3, std::make_shared<ConvolutionalBlock>, n_channels, 3, large_kernel_size, 1, false, "tanh");
}

Tensor SRResNet::forward(Tensor lr_imgs) {
  auto output = (*conv_block1)(lr_imgs);

  auto residual = output;

  output = (*residual_blocks)(output);
  output = (*conv_block2)(output);

  output = output + residual;

  output = (*subpixel_convolutional_blocks)(output);

  auto sr_imgs = (*conv_block3)(output);

  return sr_imgs;
}

}