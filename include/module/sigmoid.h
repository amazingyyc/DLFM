#ifndef NN_SIGMOID_H
#define NN_SIGMOID_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class SigmoidImpl : public ModuleImpl {
public:
  bool in_place_;

  SigmoidImpl(bool in_place = false);

  Tensor forward(Tensor) override;
};

using Sigmoid = std::shared_ptr<SigmoidImpl>;

Sigmoid sigmoid(bool in_place = false);


}


#endif //DLFM_SIGMOID_H
