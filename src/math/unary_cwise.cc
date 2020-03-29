#include "math/unary_cwise.h"

namespace dlfm {
namespace math {

template <typename T>
void assign_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec;
}

void assign(const Tensor &x, Tensor &y) {
  if (x.element_type().is<float>()) {
    assign_impl<float>(x.eigen_device().get(), x.data<float>(), y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

template <typename T>
void add_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T value, T *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec + value;
}

void add(const Tensor &x, float value, Tensor &y) {
  if (x.element_type().is<float>()) {
    add_impl<float>(x.eigen_device().get(), x.data<float>(), value, y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

template <typename T>
void sub_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T value, T *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec - value;
}

void sub(const Tensor &x, float value, Tensor &y) {
  if (x.element_type().is<float>()) {
    sub_impl<float>(x.eigen_device().get(), x.data<float>(), value, y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

template <typename T>
void multiply_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T value, T *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec * value;
}

void multiply(const Tensor &x, float value, Tensor &y) {
  if (x.element_type().is<float>()) {
    multiply_impl<float>(x.eigen_device().get(), x.data<float>(), value, y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

void divide(const Tensor &x, float value, Tensor &y) {
  multiply(x, 1.0 / value, y);
}

}
}