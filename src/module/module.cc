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

void ModuleImpl::load_torch_model(std::string model_folder) {
  for (auto m : sub_modules()) {
    m->load_torch_model(model_folder);
  }
}

std::vector<Module> ModuleImpl::sub_modules() {
  return {};
}

Tensor ModuleImpl::operator()(Tensor input) {
  return this->forward(std::move(input));
}

Tensor ModuleImpl::operator()(std::vector<Tensor> inputs) {
  return this->forward(std::move(inputs));
}


}