#ifndef NN_CARTOON_FACE_H
#define NN_CARTOON_FACE_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/conv2d.h"
#include "module/instance_norm2d.h"
#include "module/zero_pad2d.h"

namespace dlfm::nn::cartoon_face {

class ConvBlock : public ModuleImpl {
public:
  int64_t dim_out;

  Conv2d ConvBlock1;
  Conv2d ConvBlock2;
  Conv2d ConvBlock3;
  Conv2d ConvBlock4;

  ConvBlock(int64_t dim_in, int64_t dim_out);

public:
  Tensor forward(Tensor) override;
};

}
