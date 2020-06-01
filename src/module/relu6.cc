#include "module/relu6.h"

namespace dlfm::nn {

Relu6Impl::Relu6Impl(bool in_place): in_place_(in_place) {}

Tensor Relu6Impl::forward(Tensor input) {
  return input.relu6(in_place_);
}

Relu6 relu6(bool in_place) {
  return std::make_shared<Relu6Impl>(in_place);
}


}