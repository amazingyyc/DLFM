#include "module/instance_norm2d.h"

namespace dlfm::nn {

InstanceNorm2dImpl::InstanceNorm2dImpl(int64_t num_features, float eps, bool affine)
 :num_features_(num_features), eps_(eps), affine_(affine) {
  if (affine_) {
    scale_ = Tensor::create({ num_features_ });
    shift_ = Tensor::create({ num_features_ });
  }
}

void InstanceNorm2dImpl::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  if (affine_) {
    std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

    if (parent_name_scope.empty()) {
      name_scope = torch_name_scope_;
    }

    scale_.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "weight" + TORCH_MODEL_FILE_SUFFIX);
    shift_.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "bias" + TORCH_MODEL_FILE_SUFFIX);
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