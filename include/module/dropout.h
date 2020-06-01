#ifndef NN_DROPOUT_H
#define NN_DROPOUT_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class DropoutImpl : public ModuleImpl {
public:
  explicit DropoutImpl();

  Tensor forward(Tensor) override;
};

using Dropout = std::shared_ptr<DropoutImpl>;

Dropout dropout();

}

#endif