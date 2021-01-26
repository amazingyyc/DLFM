#include "module/conv_transpose2d.h"

namespace dlfm::nn {

ConvTranpose2dImpl::ConvTranpose2dImpl(int64_t in_channel,
                                       int64_t out_channel,
                                       std::vector<size_t> kernel_size,
                                       std::vector<size_t> stride,
                                       std::vector<size_t> padding,
                                       std::vector<size_t> out_padding,
                                       bool has_bias)
  : kernel_size_(std::move(kernel_size)),
    stride_(std::move(stride)),
    padding_(std::move(padding)),
    out_padding_(std::move(out_padding)),
    has_bias_(has_bias) {
  weight_ = Tensor::create({ in_channel, out_channel, (int64_t)kernel_size_[0], (int64_t)kernel_size_[1] });
  bias_ = Tensor::create({ out_channel });
}

void ConvTranpose2dImpl::load_torch_model(
    const std::unordered_map<std::string, Tensor> &tensor_map,
    std::string parent_name_scope) {
  DEF_ACTUALLY_TORCH_NAME_SCOPE;

  LOAD_TORCH_TENSOR(name_scope, "weight", weight_, tensor_map);

  if (has_bias_) {
    LOAD_TORCH_TENSOR(name_scope, "bias", bias_, tensor_map);
  } else {
    bias_.fill(0);
  }
}
Tensor ConvTranpose2dImpl::forward(Tensor input) {
  return input.conv_transpose2d(weight_, bias_, stride_, padding_, out_padding_);
}

ConvTranpose2d conv_tranpose2d(int64_t in_channel,
                               int64_t out_channel,
                               std::vector<size_t> kernel_size,
                               std::vector<size_t> stride,
                               std::vector<size_t> padding,
                               std::vector<size_t> out_padding,
                               bool has_bias) {
  return std::make_shared<ConvTranpose2dImpl>(
      in_channel,
      out_channel,
      kernel_size,
      stride,
      padding,
      out_padding,
      has_bias);
}

ConvTranpose2d conv_tranpose2d(int64_t in_channel,
                               int64_t out_channel,
                               size_t kernel_size,
                               size_t stride,
                               size_t padding,
                               size_t out_padding,
                               bool has_bias) {
  return conv_tranpose2d(in_channel,
                         out_channel,
                         { kernel_size, kernel_size },
                         { stride , stride },
                         { padding , padding },
                         { out_padding , out_padding },
                         has_bias);
}

}