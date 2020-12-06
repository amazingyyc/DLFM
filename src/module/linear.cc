#include "module/linear.h"

namespace dlfm::nn {

LinearImpl::LinearImpl(int64_t in, int64_t out, bool b): 
  in_features(in), out_features(out), use_bias(b) {
  weight = Tensor::create({out_features, in_features});

  if (use_bias) {
    bias = Tensor::create({out_features});
  }
}

void LinearImpl::load_torch_model(
  const std::unordered_map<std::string, Tensor> &tensor_map,
  std::string parent_name_scope) {
  DEF_ACTUALLY_TORCH_NAME_SCOPE;

  LOAD_TORCH_TENSOR(name_scope, "weight", weight, tensor_map);

  if (use_bias) {
    LOAD_TORCH_TENSOR(name_scope, "bias", bias, tensor_map);
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