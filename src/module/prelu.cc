#include "module/prelu.h"

namespace dlfm::nn {

PReluImpl::PReluImpl(bool in) : in_place(in) {
  w = Tensor::create({ 1 });
}

Tensor PReluImpl::forward(Tensor input) {
  return input.prelu(w, in_place);
}

PRelu prelu(bool in_place) {
  return std::make_shared<PReluImpl>(in_place);
}


}