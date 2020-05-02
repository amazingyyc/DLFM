#include "module/conv2d.h"

namespace dlfm::nn {

Conv2dImpl::Conv2dImpl(
  int64_t in_channel,
  int64_t out_channel,
  std::vector<size_t> k,
  std::vector<size_t> s,
  std::vector<size_t> p,
  bool has)
  :kernel_size(std::move(std::move(k))), stride(std::move(s)), padding(std::move(p)), has_bias(has) {
  weight = Tensor::create({ out_channel, in_channel , (int64_t)kernel_size[0], (int64_t)kernel_size[1] });
  bias = Tensor::create({ out_channel });

  if (!has_bias) {
    bias.fill(0);
  }
}

void Conv2dImpl::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scope.empty()) {
    name_scope = torch_name_scope_;
  }

  weight.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "weight" + TORCH_MODEL_FILE_SUFFIX);

  if (has_bias) {
    bias.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "bias" + TORCH_MODEL_FILE_SUFFIX);
  }
}

Tensor Conv2dImpl::forward(Tensor input) {
  return input.conv2d(weight, bias, stride, padding);
}

Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  std::vector<size_t> kernel_size,
  std::vector<size_t> stride,
  std::vector<size_t> padding,
  bool has_bias) {
  return std::make_shared<Conv2dImpl>(in_channel, out_channel, kernel_size, stride, padding, has_bias);
}

Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  size_t kernel_size,
  size_t stride,
  size_t padding,
  bool has_bias) {
  return conv2d(in_channel, out_channel, { kernel_size, kernel_size }, { stride , stride }, { padding , padding }, has_bias);
}

}