#ifndef NN_PFLD_LITE_H
#define NN_PFLD_LITE_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/conv2d.h"
#include "module/batch_norm2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::pfld_lite {

Sequential conv_bn(
  int64_t inp,
  int64_t oup,
  int64_t stride);

Sequential conv_bn1(
  int64_t inp,
  int64_t oup,
  int64_t kernel,
  int64_t stride,
  int64_t padding=1);

Sequential conv_dw(
  int64_t inp,
  int64_t oup,
  int64_t stride);

Sequential SeperableConv2d(
  int64_t in_channels,
  int64_t out_channels,
  int64_t kernel_size = 1,
  int64_t stride = 1,
  int64_t padding = 0);

class PFLDInference : public ModuleImpl {
public:
  int64_t base_channel;

  Sequential model;
  Sequential model1;

  Sequential extra;

  Conv2d conv1;
  Conv2d conv2;

  Linear fc;

  PFLDInference();

public:
  Tensor forward(Tensor) override;
};

}

#endif
