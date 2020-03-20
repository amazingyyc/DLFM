#ifndef NN_SEQUENTIAL_H
#define NN_SEQUENTIAL_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class SequentialImpl : public ModuleImpl {
public:
  std::vector<Module> modules_;

  explicit SequentialImpl(std::vector<Module>);

public:
  Module operator[](size_t);

  Tensor forward(Tensor) override;

  void torch_name_scope(std::string) override;

  std::vector<Module> sub_modules() override;
};


using Sequential = std::shared_ptr<SequentialImpl>;

Sequential sequential(std::vector<Module>);

}

#endif