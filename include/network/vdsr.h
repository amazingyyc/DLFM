#ifndef NN_VDSR_H
#define NN_VDSR_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "moduel/conv2d.h"

namespace dlfm::nn::vdsr {

class ConvReluBlock : public ModuleImpl {
public:
  Conv2d conv;
  Relu relu;

  ConvReluBlock();

public:
  Tensor forward(Tensor) override;
};

class VDSR : public ModuleImpl {
public:
  Sequential residual_layer;

  Conv2d input;
  Conv2d output;

  Relu relu;

public:
  // convert rgb[float] to ycbcr[float] [0-255]
  Tensor rgb_2_ycbcr(Tensor &rgb);

  Tensor ycbcr_2_rgb(Tensor &ycbcr);

  Tensor forward(Tensor) override;
};

}

#endif