#ifndef RESAMPLE_H
#define RESAMPLE_H

#include "basic.h"
#include "element_type.h"
#include "tensor.h"

namespace dlfm::vision {

// convert channel like RGB->BGR.
Tensor convert(const Tensor &x, std::vector<size_t> idx);

Tensor BGRA_2_RGB(const Tensor &x);

Tensor BGRA_2_RGBA(const Tensor &x)

}

#endif