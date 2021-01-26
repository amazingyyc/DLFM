#include "module/batch_norm2d.h"

namespace dlfm::nn {

BatchNorm2dImpl::BatchNorm2dImpl(int64_t num_features, float eps, bool affine, bool track_running_stats)
  :num_features_(num_features), eps_(eps), affine_(affine), track_running_stats_(track_running_stats) {
  ARGUMENT_CHECK(affine_ && track_running_stats_, "for now BatchNorm2d need affine and track_running_stats is true");

  if (affine_) {
    scale_ = Tensor::create({ num_features_ });
    shift_ = Tensor::create({ num_features_ });
  }

  if (track_running_stats_) {
    run_mean_ = Tensor::create({ num_features_ });
    run_var_ = Tensor::create({ num_features_ });
  }
}

void BatchNorm2dImpl::load_torch_model(
  const std::unordered_map<std::string, Tensor> &tensor_map,
  std::string parent_name_scope) {
  DEF_ACTUALLY_TORCH_NAME_SCOPE;

  if (affine_) {
    LOAD_TORCH_TENSOR(name_scope, "weight", scale_, tensor_map);
    LOAD_TORCH_TENSOR(name_scope, "bias", shift_, tensor_map);
  }

  if (track_running_stats_) {
    LOAD_TORCH_TENSOR(name_scope, "running_mean", run_mean_, tensor_map);
    LOAD_TORCH_TENSOR(name_scope, "running_var", run_var_, tensor_map);
  }
}

Tensor BatchNorm2dImpl::forward(Tensor x) {
  return x.batch_norm2d(run_mean_, run_var_, scale_, shift_, eps_);
}

BatchNorm2d batch_norm2d(int64_t num_features, float eps, bool affine, bool track_running_stats) {
  return std::make_shared<BatchNorm2dImpl>(num_features, eps, affine, track_running_stats);
}

}