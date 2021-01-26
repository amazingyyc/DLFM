#include "module/prelu.h"

namespace dlfm::nn {

PReluImpl::PReluImpl(bool in, int64_t num_parameter) : in_place(in) {
  w = Tensor::create({ num_parameter });
}

void PReluImpl::load_torch_model(
    const std::unordered_map<std::string, Tensor> &tensor_map,
    std::string parent_name_scope) {
  DEF_ACTUALLY_TORCH_NAME_SCOPE;

  LOAD_TORCH_TENSOR(name_scope, "weight", w, tensor_map);
}

Tensor PReluImpl::forward(Tensor input) {
  return input.prelu(w, in_place);
}

PRelu prelu(bool in_place, int64_t num_parameter) {
  return std::make_shared<PReluImpl>(in_place, num_parameter);
}


}