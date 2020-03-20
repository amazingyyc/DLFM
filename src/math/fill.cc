#include "math/fill.h"

namespace dlfm {
namespace math {

template <typename T>
void fill_impl(Eigen::ThreadPoolDevice *eigen_device, T *ptr, int64_t size, T value) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> tensor(ptr, size);

  tensor.setConstant(value);
}

void fill(Tensor &tensor, float value) {
  if (tensor.element_type().is<float>()) {
    fill_impl<float>(tensor.eigen_device().get(), tensor.data<float>(), tensor.size(), value);
  } else if (tensor.element_type().is<uint8_t>()) {
    fill_impl<uint8_t>(tensor.eigen_device().get(), tensor.data<uint8_t>(), tensor.size(), value);
  } else {
    RUNTIME_ERROR("element type:" << tensor.element_type().name() << " nor support!");
  }
}

}
}