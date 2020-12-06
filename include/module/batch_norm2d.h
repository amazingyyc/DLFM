#ifndef NN_BATCH_NORM2D_H
#define NN_BATCH_NORM2D_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class BatchNorm2dImpl : public ModuleImpl {
public:
  Tensor scale_;
  Tensor shift_;

  Tensor run_mean_;
  Tensor run_var_;

  int64_t num_features_;

  float eps_;

  bool affine_;
  bool track_running_stats_;

public:
  BatchNorm2dImpl(int64_t num_features, float eps, bool affine=true, bool track_running_stats=true);

  void load_torch_model(
    const std::unordered_map<std::string, Tensor> &tensor_map,
    std::string parent_name_scope) override;

  Tensor forward(Tensor) override;
};

using BatchNorm2d = std::shared_ptr<BatchNorm2dImpl>;

BatchNorm2d batch_norm2d(int64_t num_features, float eps = 1e-5, bool affine = true, bool track_running_stats=true);

}

#endif