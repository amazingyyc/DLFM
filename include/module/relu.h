#ifndef NN_RELU_H
#define NN_RELU_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class ReluImpl : public ModuleImpl {
public:
  bool in_place_;

  explicit ReluImpl(bool in_place = true);

  Tensor forward(Tensor) override;
};

using Relu = std::shared_ptr<ReluImpl>;

Relu relu(bool);

}

#endif