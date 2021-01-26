#ifndef NN_DCGAN_H
#define NN_DCGAN_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/prelu.h"
#include "module/conv2d.h"
#include "module/linear.h"
#include "module/tanh.h"
#include "module/batch_norm2d.h"

namespace dlfm::nn::dcgan {

class ResidualBlock : public ModuleImpl {
public:
  Conv2d conv_1;
  BatchNorm2d bn_1;

  PRelu relu;

  Conv2d conv_2;
  BatchNorm2d bn_2;

public:
  ResidualBlock(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding = 1,
    bool bias = false);

  Tensor forward(Tensor) override;
};

class SubpixelBlock : public ModuleImpl {
public:
  Conv2d conv;
  BatchNorm2d bn;
  PRelu relu;

  int64_t upscale_factor;

public:
  SubpixelBlock(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding = 1,
    bool bias = false,
    int64_t upscale_factor = 2);

  Tensor forward(Tensor) override;
};

class Generator : public ModuleImpl {
public:
  int64_t in_channels;

  Linear dense_1;
  BatchNorm2d bn_1;
  PRelu relu_1;
  Sequential residual_layer;
  BatchNorm2d bn_2;
  PRelu relu_2;
  Sequential subpixel_layer;
  Conv2d conv_1;
  Tanh tanh_1;

public:
  Generator(int64_t tag=34, int64_t residual_block_size = 16, int64_t subpixel_block_size = 3);

  Tensor forward(Tensor) override;
};


}

#endif