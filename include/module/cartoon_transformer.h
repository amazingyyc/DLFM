#ifndef NN_CARTOOON_TRANSFORMER_H
#define NN_CARTOOON_TRANSFORMER_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/conv2d.h"
#include "module/conv_transpose2d.h"
#include "module/instance_norm2d.h"

namespace dlfm::nn {

class CartoonTransformer: public ModuleImpl {
public:
  Conv2d conv01_1;
  InstanceNorm2d in01_1;

  Conv2d conv02_1;
  Conv2d conv02_2;
  InstanceNorm2d in02_1;

  Conv2d conv03_1;
  Conv2d conv03_2;
  InstanceNorm2d in03_1;

  Conv2d conv04_1;
  InstanceNorm2d in04_1;
  Conv2d conv04_2;
  InstanceNorm2d in04_2;

  Conv2d conv05_1;
  InstanceNorm2d in05_1;
  Conv2d conv05_2;
  InstanceNorm2d in05_2;

  Conv2d conv06_1;
  InstanceNorm2d in06_1;
  Conv2d conv06_2;
  InstanceNorm2d in06_2;

  Conv2d conv07_1;
  InstanceNorm2d in07_1;
  Conv2d conv07_2;
  InstanceNorm2d in07_2;

  Conv2d conv08_1;
  InstanceNorm2d in08_1;
  Conv2d conv08_2;
  InstanceNorm2d in08_2;

  Conv2d conv09_1;
  InstanceNorm2d in09_1;
  Conv2d conv09_2;
  InstanceNorm2d in09_2;

  Conv2d conv10_1;
  InstanceNorm2d in10_1;
  Conv2d conv10_2;
  InstanceNorm2d in10_2;

  Conv2d conv11_1;
  InstanceNorm2d in11_1;
  Conv2d conv11_2;
  InstanceNorm2d in11_2;

  ConvTranpose2d deconv01_1;
  Conv2d deconv01_2;
  InstanceNorm2d in12_1;

  ConvTranpose2d deconv02_1;
  Conv2d deconv02_2;
  InstanceNorm2d in13_1;

  Conv2d deconv03_1;

public:
  CartoonTransformer();

  void torch_name_scope(std::string) override;

  std::vector<Module> sub_modules() override;

  Tensor forward(Tensor) override;
};

}

#endif