#include "math/clamp.h"

namespace dlfm {
namespace math {

template <typename T>
struct ClampExpr {
  T min_value_;
  T max_value_;

  ClampExpr(T min_value, T max_value)
    : min_value_(min_value), max_value_(max_value) {
  }

  inline T operator()(T x) const {
    return (std::min)((std::max)(x, min_value_), max_value_);
  }
};

template <typename T>
void clamp_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T *y, int64_t n, T min_value, T max_value) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec.unaryExpr(ClampExpr<T>(min_value, max_value));
}

void clamp(const Tensor &x, Tensor &y, float min, float max) {
  ARGUMENT_CHECK(x.size() == y.size(), "clamp need size same");

  if (x.element_type().is<float>()) {
    clamp_impl<float>(x.eigen_device().get(), x.data<float>(), y.data<float>(), x.size(), min, max);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}


}
}