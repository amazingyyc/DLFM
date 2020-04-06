#include "math/reverse.h"

namespace dlfm::math {

template <size_t ndims, typename T>
void reverse_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  T *x,
  const Shape &xshape,
  T *y,
  const Shape &yshape,
  const std::vector<bool> &reverse) {
  Eigen::array<bool, ndims> reverse_dims;
  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = xshape[i];
    ydims[i] = yshape[i];

    reverse_dims = reverse[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);

  yvec.device(*eigen_device) = xvec.reverse(reverse_dims);
}

void reverse(const Tensor &x, Tensor &y, std::vector<bool> reverse) {
  if (x.element_type().is<float>()) {
    if (1 == x.shape().ndims()) {
      reverse_impl<1, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), reverse);
    } else if (2 == x.shape().ndims()) {
      reverse_impl<2, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), reverse);
    } else if (3 == x.shape().ndims()) {
      reverse_impl<3, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), reverse);
    } else if (4 == x.shape().ndims()) {
      reverse_impl<4, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), reverse);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else if (x.element_type().is<uint8_t >()) {
    if (1 == x.shape().ndims()) {
      reverse_impl<1, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), reverse);
    } else if (2 == x.shape().ndims()) {
      reverse_impl<2, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), reverse);
    } else if (3 == x.shape().ndims()) {
      reverse_impl<3, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), reverse);
    } else if (4 == x.shape().ndims()) {
      reverse_impl<4, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), reverse);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}


}