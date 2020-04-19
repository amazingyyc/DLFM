#ifndef NN_SEQUENTIAL_H
#define NN_SEQUENTIAL_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class SequentialImpl : public ModuleImpl {
public:
  bool print_log_;
  explicit SequentialImpl(std::vector<Module>);

public:
  Module operator[](size_t);

  Tensor forward(Tensor) override;
};

using Sequential = std::shared_ptr<SequentialImpl>;

Sequential sequential(std::vector<Module>);

}

#endif