#include "module/max_pooling2d.h"

namespace dlfm::nn {

MaxPooling2dImpl::MaxPooling2dImpl(std::vector<size_t> kernel, std::vector<size_t> stride, std::vector<size_t> padding)
  : kernel_(std::move(kernel)), stride_(std::move(stride)), padding_(std::move(padding)) {
}

Tensor MaxPooling2dImpl::forward(Tensor input) {
  return input.max_pooling2d(kernel_, stride_, padding_);
}

MaxPooling2d max_pooling2d(std::vector<size_t> kernel, std::vector<size_t> stride, std::vector<size_t> padding) {
  return std::make_shared<MaxPooling2dImpl>(kernel, stride, padding);
}

}