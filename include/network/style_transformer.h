#ifndef NN_STYLE_TRANSFORMER_H
#define NN_STYLE_TRANSFORMER_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/conv2d.h"
#include "module/instance_norm2d.h"
#include "module/zero_pad2d.h"

namespace dlfm::nn::style_transformer {

class ConvBlock : public ModuleImpl {
public:
  bool upsample;
  bool relu;

  Conv2d conv;

  InstanceNorm2d norm;

public:
  ConvBlock(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride=1,
    bool will_upsample=false,
    bool will_normalize=true,
    bool will_relu=true);

  Tensor forward(Tensor) override;
};

class ResidualBlock : public ModuleImpl {
public:
  Sequential block;

public:
  ResidualBlock(int64_t);

  Tensor forward(Tensor) override;
};

class Transformer: public ModuleImpl {
public:
  Sequential model;

public:
  Transformer();

  Tensor forward(Tensor) override;
};

}

#endif