#include "common/cost_helper.h"
#include "module/sequential.h"

namespace dlfm::nn {

SequentialImpl::SequentialImpl(std::vector<Module> modules) {
  sub_modules_ = std::move(modules);

  for (int i = 0; i < sub_modules_.size(); ++i) {
    sub_modules_[i]->torch_name_scope(std::to_string(i));
  }
}

Module SequentialImpl::operator[](size_t index) {
  return sub_modules_[index];
}

Tensor SequentialImpl::forward(Tensor input) {
  auto output = input;

  for (auto m : sub_modules_) {
    output = (*m)(output);
  }

  return output;
}

Sequential sequential(std::vector<Module> modules) {
  return std::make_shared<SequentialImpl>(modules);
}



}
