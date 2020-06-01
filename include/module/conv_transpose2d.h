#ifndef NN_CONV_TRANSPOSE2D_H
#define NN_CONV_TRANSPOSE2D_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class ConvTranpose2dImpl: public ModuleImpl {
public:
  Tensor weight_;
  Tensor bias_;

  bool has_bias_;

  std::vector<size_t> kernel_size_;
  std::vector<size_t> stride_;
  std::vector<size_t> padding_;
  std::vector<size_t> out_padding_;

  ConvTranpose2dImpl(int64_t in_channel,
                 int64_t out_channel,
                 std::vector<size_t> kernel_size,
                 std::vector<size_t> stride,
                 std::vector<size_t> padding,
                 std::vector<size_t> out_padding,
                 bool has_bias);
public:
  void load_torch_model(std::string model_folder, std::string parent_name_scope) override;

  Tensor forward(Tensor) override;
};

using ConvTranpose2d = std::shared_ptr<ConvTranpose2dImpl>;

ConvTranpose2d conv_tranpose2d(int64_t in_channel,
                               int64_t out_channel,
                               std::vector<size_t> kernel_size,
                               std::vector<size_t> stride,
                               std::vector<size_t> padding,
                               std::vector<size_t> out_padding,
                               bool has_bias);

ConvTranpose2d conv_tranpose2d(int64_t in_channel,
                               int64_t out_channel,
                               size_t kernel_size,
                               size_t stride = 1,
                               size_t padding = 0,
                               size_t out_padding = 0,
                               bool has_bias = true);

}

#endif