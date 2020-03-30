#include "math/sigmoid.h"

namespace dlfm {
namespace math {

template <typename T>
struct SigmoidExpr {
  inline T operator()(T x) const {
    return 0.5 + 0.5 * std::tanh(0.5 * x);
  }
};

template <typename T>
void sigmoid_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec.unaryExpr(SigmoidExpr<T>());
}

void sigmoid(const Tensor &x, Tensor &y) {
  if (x.element_type().is<float>()) {
    sigmoid_impl<float>(x.eigen_device().get(), x.data<float>(), y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

}
}