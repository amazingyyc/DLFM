#include "module/prelu.h"

namespace dlfm::nn {

PReluImpl::PReluImpl(bool in) : in_place(in) {
  w = Tensor::create({ 1 });
}

void PReluImpl::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scope.empty()) {
    name_scope = torch_name_scope_;
  }

  w.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "weight" + TORCH_MODEL_FILE_SUFFIX);
}

Tensor PReluImpl::forward(Tensor input) {
  return input.prelu(w, in_place);
}

PRelu prelu(bool in_place) {
  return std::make_shared<PReluImpl>(in_place);
}


}