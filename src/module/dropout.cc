#include "module/dropout.h"

namespace dlfm::nn {

DropoutImpl::DropoutImpl() {
}

Tensor DropoutImpl::forward(Tensor input) {
  return input;
}

Dropout dropout() {
  return std::make_shared<DropoutImpl>();
}

}