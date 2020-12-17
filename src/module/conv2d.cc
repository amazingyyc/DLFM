#include "module/conv2d.h"

namespace dlfm::nn {

Conv2dImpl::Conv2dImpl(
  int64_t in_channel,
  int64_t out_channel,
  std::vector<size_t> k,
  std::vector<size_t> s,
  std::vector<size_t> p,
  std::vector<size_t> d,
  size_t g,
  bool has)
  :kernel_size(std::move(k)), stride(std::move(s)), padding(std::move(p)), dilation(std::move(d)), groups(g), has_bias(has) {
  ARGUMENT_CHECK(0 == in_channel % g && 0 == out_channel % g, "conv2d parameter error");

  weight = Tensor::create({ out_channel, in_channel / (int64_t)groups, (int64_t)kernel_size[0], (int64_t)kernel_size[1] });
  bias = Tensor::create({ out_channel });
}

void Conv2dImpl::load_torch_model(
  const std::unordered_map<std::string, Tensor> &tensor_map,
  std::string parent_name_scope) {
  DEF_ACTUALLY_TORCH_NAME_SCOPE;

  LOAD_TORCH_TENSOR(name_scope, "weight", weight, tensor_map);

  if (has_bias) {
    LOAD_TORCH_TENSOR(name_scope, "bias", bias, tensor_map);
  } else {
    bias.fill(0);
  }
}

Tensor Conv2dImpl::forward(Tensor input) {
  return input.conv2d(weight, bias, stride, padding, dilation, groups);
}

Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  std::vector<size_t> kernel_size,
  std::vector<size_t> stride,
  std::vector<size_t> padding,
  std::vector<size_t> dilation,
  size_t groups,
  bool has_bias) {
  return std::make_shared<Conv2dImpl>(
    in_channel,
    out_channel,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    has_bias);
}

Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  size_t kernel_size,
  size_t stride,
  size_t padding,
  size_t dilation,
  size_t groups,
  bool has_bias) {
  return conv2d(
    in_channel,
    out_channel,
    { kernel_size, kernel_size },
    { stride , stride },
    { padding , padding },
    { dilation , dilation },
    groups,
    has_bias);
}

}