#include "module/conv2d.h"

namespace dlfm::nn {

Conv2dImpl::Conv2dImpl(
  int64_t in_channel,
  int64_t out_channel,
  std::vector<size_t> k,
  std::vector<size_t> s,
  std::vector<size_t> p,
  size_t g,
  bool has)
  :kernel_size(k), stride(s), padding(p), groups(g), has_bias(has) {
  ARGUMENT_CHECK(0 == in_channel % g && 0 == out_channel % g, "conv2d parameter error");

  weight = Tensor::create({ out_channel, in_channel / (int64_t)groups, (int64_t)kernel_size[0], (int64_t)kernel_size[1] });
  bias = Tensor::create({ out_channel });
}

void Conv2dImpl::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scope.empty()) {
    name_scope = torch_name_scope_;
  }

  weight.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "weight" + TORCH_MODEL_FILE_SUFFIX);

  if (has_bias) {
    bias.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "bias" + TORCH_MODEL_FILE_SUFFIX);
  } else {
    bias.fill(0);
  }
}

Tensor Conv2dImpl::forward(Tensor input) {
  return input.conv2d(weight, bias, stride, padding, groups);
}

Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  std::vector<size_t> kernel_size,
  std::vector<size_t> stride,
  std::vector<size_t> padding,
  size_t groups,
  bool has_bias) {
  return std::make_shared<Conv2dImpl>(
    in_channel,
    out_channel,
    kernel_size,
    stride,
    padding,
    groups,
    has_bias);
}

Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  size_t kernel_size,
  size_t stride,
  size_t padding,
  size_t groups,
  bool has_bias) {
  return conv2d(
    in_channel,
    out_channel,
    { kernel_size, kernel_size },
    { stride , stride },
    { padding , padding },
    groups,
    has_bias);
}

}