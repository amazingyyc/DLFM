#include "math/cat.h"

namespace dlfm {
namespace math {

template <size_t ndims, typename T>
void cat_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, const Shape &xshape, T *y, const Shape &yshape, T *z, const Shape &zshape, int64_t axis) {
  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;
  Eigen::array<Eigen::Index, ndims> zdims;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = xshape[i];
    ydims[i] = yshape[i];
    zdims[i] = zshape[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> zvec(z, zdims);

  zvec.device(*eigen_device) = xvec.concatenate(yvec, axis);
}

void cat(const Tensor &x, const Tensor &y, Tensor &z, int64_t axis) {
  if (x.element_type().is<float>()) {
    if (1 == x.shape().ndims()) {
      cat_impl<1, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape(), axis);
    } else if (2 == x.shape().ndims()) {
      cat_impl<2, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape(), axis);
    } else if (3 == x.shape().ndims()) {
      cat_impl<3, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape(), axis);
    } else if (4 == x.shape().ndims()) {
      cat_impl<4, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape(), axis);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else if (x.element_type().is<uint8_t>()) {
    if (1 == x.shape().ndims()) {
      cat_impl<1, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), z.data<uint8_t>(), z.shape(), axis);
    } else if (2 == x.shape().ndims()) {
      cat_impl<2, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), z.data<uint8_t>(), z.shape(), axis);
    } else if (3 == x.shape().ndims()) {
      cat_impl<3, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), z.data<uint8_t>(), z.shape(), axis);
    } else if (4 == x.shape().ndims()) {
      cat_impl<4, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), z.data<uint8_t>(), z.shape(), axis);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}
}