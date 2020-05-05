#ifndef NN_RELU6_H
#define NN_RELU6_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class Relu6Impl : public ModuleImpl {
public:
  bool in_place_;

  explicit Relu6Impl(bool in_place = true);

  Tensor forward(Tensor) override;
};

using Relu6 = std::shared_ptr<Relu6Impl>;

Relu6 relu6(bool);

}

#endif