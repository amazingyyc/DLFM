#include "module/linear.h"

namespace dlfm::nn {

LinearImpl::LinearImpl(int64_t in, int64_t out, bool b): 
  in_features(in), out_features(out), use_bias(b) {
  weight = Tensor::create({out_features, in_features});

  if (use_bias) {
    bias = Tensor::create({out_features});
  }
}

void LinearImpl::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scope.empty()) {
    name_scope = torch_name_scope_;
  }

  weight.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "weight" + TORCH_MODEL_FILE_SUFFIX);

  if (use_bias) {
    bias.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "bias" + TORCH_MODEL_FILE_SUFFIX);
  }
}

Tensor LinearImpl::forward(Tensor x) {
  if (use_bias) {
    return x.matmul(weight, false, true) + bias;
  } else {
    return x.matmul(weight, false, true);
  }
}

Linear linear(int64_t in_features, int64_t out_features, bool bias) {
  return std::make_shared<LinearImpl>(in_features, out_features, bias);
}

}