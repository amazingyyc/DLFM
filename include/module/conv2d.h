#ifndef NN_CONV2D_H
#define NN_CONV2D_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class Conv2dImpl: public ModuleImpl {
public:
  Tensor weight_;
  Tensor bias_;

  std::vector<size_t> kernel_size_;
  std::vector<size_t> stride_;
  std::vector<size_t> padding_;

  Conv2dImpl(
      int64_t in_channel,
      int64_t out_channel,
      std::vector<size_t> kernel_size,
      std::vector<size_t> stride,
      std::vector<size_t> padding);

public:
  void load_torch_model(std::string model_folder) override;

  Tensor forward(Tensor) override;

};

using Conv2d = std::shared_ptr<Conv2dImpl>;

Conv2d conv2d(
        int64_t in_channel,
        int64_t out_channel,
        std::vector<size_t> kernel_size,
        std::vector<size_t> stride,
        std::vector<size_t> padding);

}

#endif