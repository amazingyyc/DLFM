#include "module/sequential.h"

namespace dlfm::nn {

SequentialImpl::SequentialImpl(std::vector<Module> modules): modules_(std::move(modules)){
}

Module SequentialImpl::operator[](size_t index) {
  return modules_[index];
}

Tensor SequentialImpl::forward(Tensor input) {
  auto output = input;

  for (auto m : modules_) {
    output = (*m)(output);
  }

  return output;
}

void SequentialImpl::torch_name_scope(std::string name) {
  torch_name_scope_ = name;

  for (int i = 0; i < modules_.size(); ++i) {
    modules_[i]->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + std::to_string(i));
  }
}

std::vector<Module> SequentialImpl::sub_modules() {
  return modules_;
}

Sequential sequential(std::vector<Module> modules) {
  return std::make_shared<SequentialImpl>(modules);
}



}
