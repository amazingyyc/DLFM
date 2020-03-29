#include "module/upsample2d.h"

namespace dlfm::nn {

Upsample2dImpl::Upsample2dImpl(float scale_factor, std::string mode, bool align_corners)
  :scale_factor_(scale_factor), mode_(mode), align_corners_(align_corners) {
}

Tensor Upsample2dImpl::forward(Tensor input) {
  return input.upsample2d(scale_factor_, mode_, align_corners_);
}

Upsample2d upsample2d(float scale_factor, std::string mode, bool align_corners) {
  return std::make_shared<Upsample2dImpl>(scale_factor, mode, align_corners);
}

}