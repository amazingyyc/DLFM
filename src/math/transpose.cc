#include "math/transpose.h"

namespace dlfm {
namespace math {

template <size_t ndims, typename T>
void transpose_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, const Shape &xshape, T *y, const Shape &yshape, std::vector<size_t> axis) {
  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;
  Eigen::array<Eigen::Index, ndims> shuffle_axis;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = xshape[i];
    ydims[i] = yshape[i];
    shuffle_axis[i] = axis[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);

  yvec.device(*eigen_device) = xvec.shuffle(shuffle_axis);
}

void transpose(const Tensor &x, Tensor &y, std::vector<size_t> axis) {
  if (x.element_type().is<float>()) {
    if (1 == x.shape().ndims()) {
      transpose_impl<1, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), axis);
    } else if (2 == x.shape().ndims()) {
      transpose_impl<2, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), axis);
    } else if (3 == x.shape().ndims()) {
      transpose_impl<3, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), axis);
    } else if (4 == x.shape().ndims()) {
      transpose_impl<4, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), axis);
    } else if (5 == x.shape().ndims()) {
      transpose_impl<5, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), axis);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else if (x.element_type().is<uint8_t >()) {
    if (1 == x.shape().ndims()) {
      transpose_impl<1, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), axis);
    } else if (2 == x.shape().ndims()) {
      transpose_impl<2, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), axis);
    } else if (3 == x.shape().ndims()) {
      transpose_impl<3, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), axis);
    } else if (4 == x.shape().ndims()) {
      transpose_impl<4, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), axis);
    } else if (5 == x.shape().ndims()) {
      transpose_impl<5, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), axis);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}


}
}