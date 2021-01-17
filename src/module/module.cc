#include "common/deserialize.h"
#include "module/module.h"

namespace dlfm::nn {

ModuleImpl::ModuleImpl() = default;

Tensor ModuleImpl::forward(Tensor) {
    RUNTIME_ERROR("this module does not implement ModuleImpl::forward(Tensor)");
}

Tensor ModuleImpl::forward(std::vector<Tensor>) {
    RUNTIME_ERROR("this module does not implement ModuleImpl::forward(std::vector<Tensor>)");
}

Tensor ModuleImpl::operator()(Tensor input) {
  return this->forward(input);
}

Tensor ModuleImpl::operator()(std::vector<Tensor> inputs) {
  return this->forward(inputs);
}

void ModuleImpl::torch_name_scope(std::string name) {
  torch_name_scope_ = name;
}

void ModuleImpl::load_torch_model(const std::string &file_path, const std::string &name_scope) {
  ModelDeserialize deserialize(file_path);

  std::unordered_map<std::string, Tensor> tensor_dict;
  deserialize.deserialize(tensor_dict);

  load_torch_model(tensor_dict, name_scope);
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


}