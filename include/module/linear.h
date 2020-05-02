#ifndef NN_LINEAR_H
#define NN_LINEAR_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class LinearImpl : public ModuleImpl {
public:
  int64_t in_features;
  int64_t out_features;
  bool use_bias;

  Tensor weight;
  Tensor bias;

  LinearImpl(int64_t in_features, int64_t out_features, bool bias=true);

public:
  void load_torch_model(std::string model_folder, std::string parent_name_scope) override;

  Tensor forward(Tensor) override;
};

using Linear = std::shared_ptr<LinearImpl>;

Linear linear(int64_t in_features, int64_t out_features, bool bias=true);

}

#endif