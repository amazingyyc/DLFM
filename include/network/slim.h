#ifndef NN_SLIM_H
#define NN_SLIM_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/conv2d.h"
#include "module/batch_norm2d.h"
#include "module/linear.h"

namespace dlfm::nn::slim {

Sequential conv_bn(int64_t inp, int64_t oup, int64_t stride=1);

Sequential depth_conv2d(int64_t inp, int64_t oup, int64_t kernel=1, int64_t stride=1, int64_t pad=0);

Sequential conv_dw(int64_t inp, int64_t oup, int64_t stride, int64_t padding=1);

class Slim: public ModuleImpl {
public:
  int64_t num_classes;

  Sequential conv1;
  Sequential conv2;
  Sequential conv3;
  Sequential conv4;
  Sequential conv5;
  Sequential conv6;
  Sequential conv7;
  Sequential conv8;
  Sequential conv9;
  Sequential conv10;
  Sequential conv11;
  Sequential conv12;
  Sequential conv13;

  Linear fc;

public:
  Slim();

  Tensor forward(Tensor) override;
};

}

#endif