#ifndef NN_CARTOOON_TRANSFORMER_H
#define NN_CARTOOON_TRANSFORMER_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/conv2d.h"
#include "module/conv_transpose2d.h"
#include "module/instance_norm2d.h"

namespace dlfm::nn::cartoon_transformer {

class InstanceNormalization: public ModuleImpl {
public:
  int64_t num_features_;
  
  float eps_;

  Tensor scale_;
  Tensor shift_;

public:
  InstanceNormalization(int64_t num_features, float eps=1e-9);

  void load_torch_model(std::string model_folder, std::string parent_name_scope) override;

  Tensor forward(Tensor) override;
};

class CartoonTransformer: public ModuleImpl {
public:
  Conv2d conv01_1;
  std::shared_ptr<InstanceNormalization> in01_1;

  Conv2d conv02_1;
  Conv2d conv02_2;
  std::shared_ptr<InstanceNormalization> in02_1;

  Conv2d conv03_1;
  Conv2d conv03_2;
  std::shared_ptr<InstanceNormalization> in03_1;

  Conv2d conv04_1;
  std::shared_ptr<InstanceNormalization> in04_1;
  Conv2d conv04_2;
  std::shared_ptr<InstanceNormalization> in04_2;

  Conv2d conv05_1;
  std::shared_ptr<InstanceNormalization> in05_1;
  Conv2d conv05_2;
  std::shared_ptr<InstanceNormalization> in05_2;

  Conv2d conv06_1;
  std::shared_ptr<InstanceNormalization> in06_1;
  Conv2d conv06_2;
  std::shared_ptr<InstanceNormalization> in06_2;

  Conv2d conv07_1;
  std::shared_ptr<InstanceNormalization> in07_1;
  Conv2d conv07_2;
  std::shared_ptr<InstanceNormalization> in07_2;

  Conv2d conv08_1;
  std::shared_ptr<InstanceNormalization> in08_1;
  Conv2d conv08_2;
  std::shared_ptr<InstanceNormalization> in08_2;

  Conv2d conv09_1;
  std::shared_ptr<InstanceNormalization> in09_1;
  Conv2d conv09_2;
  std::shared_ptr<InstanceNormalization> in09_2;

  Conv2d conv10_1;
  std::shared_ptr<InstanceNormalization> in10_1;
  Conv2d conv10_2;
  std::shared_ptr<InstanceNormalization> in10_2;

  Conv2d conv11_1;
  std::shared_ptr<InstanceNormalization> in11_1;
  Conv2d conv11_2;
  std::shared_ptr<InstanceNormalization> in11_2;

  ConvTranpose2d deconv01_1;
  Conv2d deconv01_2;
  std::shared_ptr<InstanceNormalization> in12_1;

  ConvTranpose2d deconv02_1;
  Conv2d deconv02_2;
  std::shared_ptr<InstanceNormalization> in13_1;

  Conv2d deconv03_1;

public:
  CartoonTransformer();

  Tensor forward(Tensor) override;

  Tensor test(Tensor);
};

}

#endif