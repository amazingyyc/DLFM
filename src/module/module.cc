#include "module/module.h"

#include <utility>

namespace dlfm::nn {

ModuleImpl::ModuleImpl() = default;

Tensor ModuleImpl::forward(Tensor) {
    RUNTIME_ERROR("this module does not implement ModuleImpl::forward(Tensor)");
}

Tensor ModuleImpl::forward(std::vector<Tensor>) {
    RUNTIME_ERROR("this module does not implement ModuleImpl::forward(std::vector<Tensor>)");
}

void ModuleImpl::torch_name_scope(std::string name) {
  torch_name_scope_ = name;
}

void ModuleImpl::load_torch_model(std::string model_folder, std::string parent_name_scop) {
  auto name_scope = parent_name_scop + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scop.empty()) {
    name_scope = torch_name_scope_;
  }

  for (auto m : sub_modules()) {
    m->load_torch_model(model_folder, name_scope);
  }
}

std::vector<Module> ModuleImpl::sub_modules() {
  return sub_modules_;;
}

Tensor ModuleImpl::operator()(Tensor input) {
  return this->forward(std::move(input));
}

Tensor ModuleImpl::operator()(std::vector<Tensor> inputs) {
  return this->forward(std::move(inputs));
}


}