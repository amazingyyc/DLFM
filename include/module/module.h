#ifndef NN_MODULE_H
#define NN_MODULE_H

#include "common/tensor.h"

namespace dlfm::nn {

#define TORCH_NAME_SCOPE_SEP "."
#define TORCH_MODEL_FILE_SUFFIX ".bin"

class ModuleImpl {
protected:
  // model name scope (used for loading pytorh model)
  std::string torch_name_scope_;

  ModuleImpl();

  virtual Tensor forward(Tensor);

  virtual Tensor forward(std::vector<Tensor>);

public:
  // special function for pytorch
  virtual void torch_name_scope(std::string name);

  // load torch model
  virtual void load_torch_model(std::string model_folder);

  virtual std::vector<std::shared_ptr<ModuleImpl>> sub_modules();

  Tensor operator()(Tensor);

  Tensor operator()(std::vector<Tensor>);

};

using Module = std::shared_ptr<ModuleImpl>;

}

#endif // !MODULE_H
