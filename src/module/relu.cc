#include "module/relu.h"

namespace dlfm::nn {

ReluImpl::ReluImpl(bool in_place): in_place_(in_place) {}

Tensor ReluImpl::forward(Tensor input) {
  return input.relu(in_place_);
}

Relu relu(bool in_place) {
  return std::make_shared<ReluImpl>(in_place);
}


}