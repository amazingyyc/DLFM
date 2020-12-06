#ifndef NN_MODULE_H
#define NN_MODULE_H

#include "common/tensor.h"

namespace dlfm::nn {

#define TORCH_NAME_SCOPE_SEP "."
#define TORCH_MODEL_FILE_SUFFIX ".bin"

#define ADD_SUB_MODULE(variable, builder, ...) \
{ \
  variable = builder(__VA_ARGS__); \
  sub_modules_.emplace_back(variable); \
  variable->torch_name_scope(#variable); \
} \

// find tensor from tensor_map
#define LOAD_TORCH_TENSOR(scope, name, tensor, tensor_map) \
{ \
  auto tensor_name = scope + TORCH_NAME_SCOPE_SEP + name; \
  if (scope.empty()) { \
    tensor_name = name; \
  } \
  auto it = tensor_map.find(tensor_name); \
  ARGUMENT_CHECK(it != tensor_map.end(), "Can not find tensor:" << tensor_name); \
  tensor = it->second; \
} \

// define the actually name scope
#define DEF_ACTUALLY_TORCH_NAME_SCOPE \
auto name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_; \
if (parent_name_scope.empty()) { \
  name_scope = torch_name_scope_; \
} \


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
  virtual void load_torch_model(
    const std::unordered_map<std::string, Tensor> &tensor_map,
    std::string parent_name_scope = "");

  virtual std::vector<std::shared_ptr<ModuleImpl>> sub_modules();

  Tensor operator()(Tensor);

  Tensor operator()(std::vector<Tensor>);

};

using Module = std::shared_ptr<ModuleImpl>;

}

#endif // !MODULE_H
