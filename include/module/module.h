#ifndef NN_MODULE_H
#define NN_MODULE_H

#include "common/tensor.h"

namespace dlfm::nn {

#define TORCH_NAME_SCOPE_SEP "."
#define TORCH_MODEL_FILE_SUFFIX ".bin"

#define ADD_SUB_MODULE(variable, builder, ...)     \
{                                                  \
  variable = builder(__VA_ARGS__);                 \
  sub_modules_.emplace_back(variable);             \
  variable->torch_name_scope(#variable);           \
}                                                  \

class ModuleImpl {
public:
  // model name scope (used for loading pytorh model)
  std::string torch_name_scope_;

  // sub modules
  std::vector<std::shared_ptr<ModuleImpl>> sub_modules_;

  ModuleImpl();

  virtual Tensor forward(Tensor);

  virtual Tensor forward(std::vector<Tensor>);

public:
  // special function for pytorch
  void torch_name_scope(std::string name);

  // load torch model
  virtual void load_torch_model(std::string model_folder, std::string parent_name_scop = "");

  virtual std::vector<std::shared_ptr<ModuleImpl>> sub_modules();

  Tensor operator()(Tensor);

  Tensor operator()(std::vector<Tensor>);

};

using Module = std::shared_ptr<ModuleImpl>;

}

#endif // !MODULE_H
