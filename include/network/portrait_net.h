#ifndef NN_PORTRAIT_NET_H
#define NN_PORTRAIT_NET_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/conv2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::portrait_net {

class InvertedResidual: public ModuleImpl {
public:
  int64_t stride;

  bool use_res_connect;

  Sequential conv;

  InvertedResidual(int64_t inp, int64_t oup, int64_t stride, int64_t expand_ratio, int64_t dilation=1);

public:
  Tensor forward(Tensor) override;
};

class ResidualBlock: public ModuleImpl {
public:
  Sequential block;

  Sequential residual;

  ResidualBlock(int64_t inp, int64_t oup, int64_t stride=1);

public:
  Sequential conv_dw(int64_t inp, int64_t oup, int64_t kernel, int64_t stride);

  Tensor forward(Tensor) override;
};

class MobileNetV2: public ModuleImpl {
public:
  bool add_edge;

  float channel_ratio;

  int64_t min_channel;

  bool use_deconv_group;

  Sequential stage0;

  std::shared_ptr<InvertedResidual> stage1;

  Sequential stage2;
  Sequential stage3;
  Sequential stage4;
  Sequential stage5;
  Sequential stage6;

  std::shared_ptr<InvertedResidual> stage7;

  Module deconv1;
  Module deconv2;
  Module deconv3;
  Module deconv4;
  Module deconv5;

  std::shared_ptr<ResidualBlock> transit1;
  std::shared_ptr<ResidualBlock> transit2;
  std::shared_ptr<ResidualBlock> transit3;
  std::shared_ptr<ResidualBlock> transit4;
  std::shared_ptr<ResidualBlock> transit5;

  Conv2d pred;
  Conv2d edge;

  MobileNetV2(
          int64_t n_class=2,
          bool useUpsample=false,
          bool useDeconvGroup=false,
          bool addEdge=false,
          float channelRatio=1.0,
          int64_t minChannel=16,
          bool video=false);

public:
  int64_t depth(int64_t channels);

  Sequential conv_bn(int64_t inp, int64_t oup, int64_t kernel, int64_t stride);

  Tensor forward(Tensor) override;
};

}

#endif