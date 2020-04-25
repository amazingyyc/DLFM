#include "module/tanh.h"

namespace dlfm::nn {

TanhImpl::TanhImpl(bool in_place): in_place_(in_place) {}

Tensor TanhImpl::forward(Tensor input) {
  return input.tanh(in_place_);
}

Tanh tanh(bool in_place) {
  return std::make_shared<TanhImpl>(in_place);
}


}