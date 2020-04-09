#ifndef NN_REFLECTION_PAD2D_H
#define NN_REFLECTION_PAD2D_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class ReflectionPad2dImpl : public ModuleImpl {
public:
  std::vector<size_t> paddings_;

  explicit ReflectionPad2dImpl(size_t padding);
  explicit ReflectionPad2dImpl(std::vector<size_t> paddings);

  Tensor forward(Tensor) override;
};

using ReflectionPad2d = std::shared_ptr<ReflectionPad2dImpl>;

ReflectionPad2d reflection_pad2d(size_t padding);
ReflectionPad2d reflection_pad2d(std::vector<size_t> paddings);

}

#endif