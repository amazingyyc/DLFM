#include "module/sigmoid.h"

namespace dlfm::nn {

SigmoidImpl::SigmoidImpl(bool in_place): in_place_(in_place) {
}

Tensor SigmoidImpl::forward(Tensor input) {
  return input.sigmoid(in_place_);
}

Sigmoid sigmoid(bool in_place) {
  return std::make_shared<SigmoidImpl>(in_place);
}


}