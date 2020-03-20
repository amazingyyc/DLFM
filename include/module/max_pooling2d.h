#ifndef NN_MAX_POOLING_H
#define NN_MAX_POOLING_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class MaxPooling2dImpl : public ModuleImpl {
public:
  std::vector<size_t> kernel_;
  std::vector<size_t> stride_;
  std::vector<size_t> padding_;

  MaxPooling2dImpl(std::vector<size_t> kernel, std::vector<size_t> stride, std::vector<size_t> padding);

public:
  Tensor forward(Tensor) override;
};

using MaxPooling2d = std::shared_ptr<MaxPooling2dImpl>;

MaxPooling2d max_pooling2d(std::vector<size_t> kernel, std::vector<size_t> stride, std::vector<size_t> padding);

}

#endif