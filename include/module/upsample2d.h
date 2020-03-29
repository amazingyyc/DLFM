#ifndef NN_UPSAMPLE2D_H
#define NN_UPSAMPLE2D_H

#include "common/tensor.h"
#include "module/module.h"

namespace dlfm::nn {

class Upsample2dImpl : public ModuleImpl {
public:
  float scale_factor_;

  std::string mode_;

  bool align_corners_;

  Upsample2dImpl(float scale_factor, std::string mode = "nearest", bool align_corners = false);

  Tensor forward(Tensor) override;
};

using Upsample2d = std::shared_ptr<Upsample2dImpl>;

Upsample2d upsample2d(float scale_factor, std::string mode = "nearest", bool align_corners = false);

}
#endif //DLFM_UPSAMPLE_H
