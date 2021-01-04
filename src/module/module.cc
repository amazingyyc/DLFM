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

void ModuleImpl::load_torch_model(
    const std::unordered_map<std::string, Tensor> &tensor_map,
    std::string parent_name_scope) {
  DEF_ACTUALLY_TORCH_NAME_SCOPE;

  for (auto m : sub_modules()) {
    m->load_torch_model(tensor_map, name_scope);
  }
}

std::vector<Module> ModuleImpl::sub_modules() {
  return sub_modules_;
}

Tensor ModuleImpl::operator()(Tensor input) {
  return this->forward(input);
}

Tensor ModuleImpl::operator()(std::vector<Tensor> inputs) {
  return this->forward(inputs);
}


}