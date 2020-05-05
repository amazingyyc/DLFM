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

void ConvTranpose2dImpl::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scope.empty()) {
    name_scope = torch_name_scope_;
  }

  weight_.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "weight" + TORCH_MODEL_FILE_SUFFIX);

  if (has_bias_) {
    bias_.initialize_from_file(
            model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "bias" + TORCH_MODEL_FILE_SUFFIX);
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