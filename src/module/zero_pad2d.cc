#include "module/zero_pad2d.h"

namespace dlfm::nn {

ZeroPad2dImpl::ZeroPad2dImpl(size_t padding)
    : ZeroPad2dImpl({padding, padding, padding, padding}) {}

ZeroPad2dImpl::ZeroPad2dImpl(std::vector<size_t> padding) : padding_(padding) {
}

Tensor ZeroPad2dImpl::forward(Tensor x) { 
  return x.pad(padding_); 
}

ZeroPad2d zero_pad2d(size_t padding) {
  return zero_pad2d({padding, padding, padding, padding});
}

ZeroPad2d zero_pad2d(std::vector<size_t> padding) {
  return std::make_shared<ZeroPad2dImpl>(padding);
}

}