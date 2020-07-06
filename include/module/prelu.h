#ifndef NN_PRELU_H
#define NN_PRELU_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class PReluImpl : public ModuleImpl {
public:
  bool in_place;

  Tensor w;

  explicit PReluImpl(bool in_place, int64_t num_parameter=1);

public:
  void load_torch_model(std::string model_folder, std::string parent_name_scope) override;
  
  Tensor forward(Tensor) override;
};

using PRelu = std::shared_ptr<PReluImpl>;

PRelu prelu(bool, int64_t num_parameter=1);

}

#endif