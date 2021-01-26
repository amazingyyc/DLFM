#include "module/instance_norm2d.h"

namespace dlfm::nn {

InstanceNorm2dImpl::InstanceNorm2dImpl(int64_t num_features, float eps, bool affine)
 :num_features_(num_features), eps_(eps), affine_(affine) {
  if (affine_) {
    scale_ = Tensor::create({ num_features_ });
    shift_ = Tensor::create({ num_features_ });
  }
}

void InstanceNorm2dImpl::load_torch_model(
  const std::unordered_map<std::string, Tensor> &tensor_map,
  std::string parent_name_scope) {
  if (affine_) {
    DEF_ACTUALLY_TORCH_NAME_SCOPE;

    LOAD_TORCH_TENSOR(name_scope, "weight", scale_, tensor_map);
    LOAD_TORCH_TENSOR(name_scope, "bias", shift_, tensor_map);
  }
}

Tensor InstanceNorm2dImpl::forward(Tensor input) {
  if (affine_) {
    return input.instance_norm2d(scale_, shift_, eps_);
  } else {
    return input.instance_norm2d(eps_);
  }
}

InstanceNorm2d instance_norm2d(int64_t num_features, float eps, bool affine) {
  return std::make_shared<InstanceNorm2dImpl>(num_features, eps, affine);
}

}