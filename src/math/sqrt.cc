#include "math/sqrt.h"

namespace dlfm {
namespace math {

template <typename T>
void sqrt_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec.sqrt();
}

void sqrt(const Tensor &x, Tensor &y) {
  if (x.element_type().is<float>()) {
    sqrt_impl<float>(x.eigen_device().get(), x.data<float>(), y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

}
}