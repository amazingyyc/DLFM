#include "common/cost_helper.h"
#include "module/sequential.h"

namespace dlfm::nn {

SequentialImpl::SequentialImpl(std::vector<Module> modules) {
  sub_modules_ = std::move(modules);

  for (int i = 0; i < sub_modules_.size(); ++i) {
    sub_modules_[i]->torch_name_scope(std::to_string(i));
  }

  print_log_ = false;
}

Module SequentialImpl::operator[](size_t index) {
  return sub_modules_[index];
}

Tensor SequentialImpl::forward(Tensor input) {
  auto output = input;

  for (auto m : sub_modules_) {
    if (print_log_) {
      CostHelper::start(m->torch_name_scope_);
    }

    output = (*m)(output);

    if (print_log_) {
      CostHelper::end();
    }
  }

  return output;
}

Sequential sequential(std::vector<Module> modules) {
  return std::make_shared<SequentialImpl>(modules);
}



}
