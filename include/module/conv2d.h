#ifndef NN_CONV2D_H
#define NN_CONV2D_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class Conv2dImpl: public ModuleImpl {
public:
  Tensor weight;
  Tensor bias;

  std::vector<size_t> kernel_size;
  std::vector<size_t> stride;
  std::vector<size_t> padding;
  std::vector<size_t> dilation;

  size_t groups;

  bool has_bias;

public:
  Conv2dImpl(
    int64_t in_channel,
    int64_t out_channel,
    const std::vector<size_t> &kernel_size,
    const std::vector<size_t> &stride,
    const std::vector<size_t> &padding,
    const std::vector<size_t> &dilation,
    size_t groups,
    bool has_bias);

public:
  virtual void load_torch_model(
    const std::unordered_map<std::string, Tensor> &tensor_map,
    std::string parent_name_scop) override;

  Tensor forward(Tensor) override;

};

using Conv2d = std::shared_ptr<Conv2dImpl>;

Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  const std::vector<size_t> &kernel_size,
  const std::vector<size_t> &stride,
  const std::vector<size_t> &padding,
  const std::vector<size_t> &dilation,
  size_t groups,
  bool has_bias);

Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  size_t kernel_size,
  size_t stride,
  size_t padding,
  size_t dilation,
  size_t groups,
  bool has_bias);
}

#endif