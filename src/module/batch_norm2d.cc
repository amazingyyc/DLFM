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

void BatchNorm2dImpl::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scope.empty()) {
    name_scope = torch_name_scope_;
  }

  if (affine_) {
    scale_.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "weight" + TORCH_MODEL_FILE_SUFFIX);
    shift_.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "bias" + TORCH_MODEL_FILE_SUFFIX);
  }

  if (track_running_stats_) {
    run_mean_.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "running_mean" + TORCH_MODEL_FILE_SUFFIX);
    run_var_.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "running_var" + TORCH_MODEL_FILE_SUFFIX);
  }
}

Tensor BatchNorm2dImpl::forward(Tensor x) {
  return x.batch_norm2d(run_mean_, run_var_, scale_, shift_, eps_);
}

BatchNorm2d batch_norm2d(int64_t num_features, float eps, bool affine, bool track_running_stats) {
  return std::make_shared<BatchNorm2dImpl>(num_features, eps, affine, track_running_stats);
}

}