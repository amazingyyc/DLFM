#ifndef NN_SRRESNET_H
#define NN_SRRESNET_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/conv2d.h"
#include "module/relu6.h"

namespace dlfm::nn::srresnet {

class ResNet : public ModuleImpl {
public:
  Conv2d conv1;
  Relu6 relu1;
  Conv2d conv2;
  Relu6 relu2;

  ResNet(int64_t in_channels = 32, int64_t out_channels = 32, int64_t kernel_size = 3, int64_t stride = 1, int64_t padding = 1);

public:
  Tensor forward(Tensor) override;
};

class SRResNet: public ModuleImpl {
public:
  Sequential input;

  std::shared_ptr<ResNet> res1;
  std::shared_ptr<ResNet> res2;

  Sequential output;

  SRResNet(int64_t in_channels = 3, int64_t out_channels = 3);

public:
  Tensor forward(Tensor) override;
};

}

#endif