#include "module/reflection_pad2d.h"

namespace dlfm::nn {

ReflectionPad2dImpl::ReflectionPad2dImpl(size_t padding): ReflectionPad2dImpl({ padding , padding , padding , padding }) {

}

ReflectionPad2dImpl::ReflectionPad2dImpl(std::vector<size_t> paddings): paddings_(paddings) {
  ARGUMENT_CHECK(4 == paddings_.size(), "ReflectionPad2dImpl need paddings size is 4");
}

Tensor ReflectionPad2dImpl::forward(Tensor x) {
  return x.reflection_pad2d(paddings_);
}

ReflectionPad2d reflection_pad2d(size_t padding) {
  return std::make_shared<ReflectionPad2dImpl>(padding);
}

ReflectionPad2d reflection_pad2d(std::vector<size_t> paddings) {
  return std::make_shared<ReflectionPad2dImpl>(paddings);
}

}