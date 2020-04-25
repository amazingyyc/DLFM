#ifndef NN_TANH_H
#define NN_TANH_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class TanhImpl : public ModuleImpl {
public:
  bool in_place_;

  explicit TanhImpl(bool in_place = true);

  Tensor forward(Tensor) override;
};

using Tanh = std::shared_ptr<TanhImpl>;

Tanh tanh(bool);

}

#endif