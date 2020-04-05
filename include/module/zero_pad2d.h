#ifndef NN_PAD_H
#define NN_PAD_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class ZeroPad2dImpl : public ModuleImpl {
public:
  std::vector<size_t> padding_;

  ZeroPad2dImpl(size_t padding);

  ZeroPad2dImpl(std::vector<size_t> padding);

public:
  Tensor forward(Tensor) override;
};

using ZeroPad2d = std::shared_ptr<ZeroPad2dImpl>;

ZeroPad2d zero_pad2d(size_t padding);

ZeroPad2d zero_pad2d(std::vector<size_t> padding);

} // namespace dlfm::nn


#endif