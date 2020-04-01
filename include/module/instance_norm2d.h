#ifndef NN_INSTANCENORM2D_H
#define NN_INSTANCENORM2D_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class InstanceNorm2dImpl : public ModuleImpl {
public:
  Tensor scale_;
  Tensor shift_;

  int64_t num_features_;

  float eps_;

  bool affine_;

public:
  InstanceNorm2dImpl(int64_t num_features, float eps, bool affine);

  void load_torch_model(std::string model_folder) override;

  Tensor forward(Tensor) override;
};

using InstanceNorm2d = std::shared_ptr<InstanceNorm2dImpl>;

InstanceNorm2d instance_norm2d(int64_t num_features, float eps, bool affine);

}

#endif