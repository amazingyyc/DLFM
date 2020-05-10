#ifndef NN_PRELU_H
#define NN_PRELU_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class PReluImpl : public ModuleImpl {
public:
  bool in_place;

  Tensor w;

  explicit PReluImpl(bool in_place);

public:
  Tensor forward(Tensor) override;
};

using PRelu = std::shared_ptr<PReluImpl>;

PRelu prelu(bool);

}

#endif