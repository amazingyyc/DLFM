#ifndef NN_SRGAN_H
#define NN_SRGAN_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/conv2d.h"
#include "module/sequential.h"

namespace dlfm::nn::srgan {

class ConvolutionalBlock : public ModuleImpl {
public:
  Sequential conv_block;

  ConvolutionalBlock(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride=1, bool batch_norm=false, std::string activation="");

public:
  Tensor forward(Tensor) override;
};

class SubPixelConvolutionalBlock: public ModuleImpl {
public:
  Conv2d conv;

  int64_t scaling_factor;

  PRelu prelu;

  SubPixelConvolutionalBlock(int64_t kernel_size=3, int64_t n_channels=64, int64_t scaling_factor=2);

public:
  Tensor forward(Tensor) override;
};

class ResidualBlock : public ModuleImpl {
public:
  std::shared_ptr<ConvolutionalBlock> conv_block1;
  std::shared_ptr<ConvolutionalBlock> conv_block2;

  ResidualBlock(int64_t kernel_size=3, int64_t n_channels=64);

public:
  Tensor forward(Tensor) override;
};

class SRResNet : public ModuleImpl {
public:
  std::shared_ptr<ConvolutionalBlock> conv_block1;

  Sequential residual_blocks;

  std::shared_ptr<ConvolutionalBlock> conv_block2;

  Sequential subpixel_convolutional_blocks;

  std::shared_ptr<ConvolutionalBlock> conv_block3;

  SRResNet(int64_t large_kernel_size=9, int64_t small_kernel_size=3, int64_t n_channels=64, int64_t n_blocks=16, int64_t scaling_factor=4);

public:
  Tensor forward(Tensor) override;
};

}