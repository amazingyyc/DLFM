#include "module/conv2d.h"

namespace dlfm::nn {

Conv2dImpl::Conv2dImpl(int64_t in_channel,
                     int64_t out_channel,
                     std::vector<size_t> kernel_size,
                     std::vector<size_t> stride,
                     std::vector<size_t> padding)
  :kernel_size_(std::move(std::move(kernel_size))), stride_(std::move(stride)), padding_(std::move(padding)) {
  weight_ = Tensor::create({ out_channel, in_channel , (int64_t)kernel_size_[0], (int64_t)kernel_size_[1] });
  bias_   = Tensor::create({ out_channel});
}

void Conv2dImpl::load_torch_model(std::string model_folder) {
  weight_.initialize_from_file(model_folder + FILE_SEP + torch_name_scope_ + TORCH_NAME_SCOPE_SEP + "weight" + TORCH_MODEL_FILE_SUFFIX);
  bias_.initialize_from_file(model_folder + FILE_SEP + torch_name_scope_ + TORCH_NAME_SCOPE_SEP + "bias" + TORCH_MODEL_FILE_SUFFIX);
}

Tensor Conv2dImpl::forward(Tensor input) {
  return input.conv2d(weight_, bias_, stride_, padding_);
}

Conv2d conv2d(
        int64_t in_channel,
        int64_t out_channel,
        std::vector<size_t> kernel_size,
        std::vector<size_t> stride,
        std::vector<size_t> padding) {
  return std::make_shared<Conv2dImpl>(in_channel, out_channel, kernel_size, stride, padding);
}

Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  size_t kernel_size,
  size_t stride,
  size_t padding) {
  return conv2d(in_channel, out_channel, { kernel_size, kernel_size }, { stride , stride }, { padding , padding });
}

}