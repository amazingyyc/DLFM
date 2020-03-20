#include "math/cast.h"

namespace dlfm {
namespace math {

template <typename From, typename To>
void cast_impl(Eigen::ThreadPoolDevice *eigen_device, From *x, To *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<From, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<To, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec.template cast<To>();
}

void cast(const Tensor &x, Tensor &y) {
  auto xtype = x.element_type();
  auto ytype = y.element_type();

  if (xtype.is<float>() && ytype.is<uint8_t>()) {
    cast_impl<float, uint8_t>(x.eigen_device().get(), x.data<float>(), y.data<uint8_t>(), x.size());
  } else if (xtype.is<uint8_t>() && ytype.is<float>()) {
    cast_impl<uint8_t, float>(x.eigen_device().get(), x.data<uint8_t>(), y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("cast from:" << xtype.name() << " to:" << ytype.name() << " not support.");
  }
}

}
}