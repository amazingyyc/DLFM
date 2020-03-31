#include "math/mean.h"

namespace dlfm {
namespace math {

template <typename T, int ndims, int reduce_count>
void mean_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, const Shape &xshape, T *y, const Shape &yshape) {
  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = xshape[i];
    ydims[i] = yshape[i];
  }

  Eigen::array<Eigen::Index, reduce_count> reduce_dims;

  for (int i = 0, j = 0; i < ndims; ++i) {
    if (xshape[i] != yshape[i]) {
      reduce_dims[j++] = i;
    }
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);

  yvec.device(*eigen_device) = xvec.mean(reduce_dims).reshape(ydims);
}

void mean(const Tensor &x, Tensor &y) {
  ARGUMENT_CHECK(x.element_type().is<float>(), "only support float");
  ARGUMENT_CHECK(x.shape().rank() == y.shape().rank(), "mean need shape rank is same");

  int ndims = x.shape().rank();
  int reduce_count = 0;

  for (int i = 0; i < ndims; ++i) {
    if (x.shape()[i] != y.shape()[i]) {
      reduce_count += 1;
    }

    ARGUMENT_CHECK(x.shape()[i] == y.shape()[i] || 1 == y.shape()[i], "shape error");
  }

  if (1 == ndims) {
    if (1 == reduce_count) {
      mean_impl<float, 1, 1>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
    } else {
      RUNTIME_ERROR("the shape is error");
    }
  } else if (2 == ndims) {
    if (1 == reduce_count) {
      mean_impl<float, 2, 1>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
    } else if (2 == reduce_count) {
      mean_impl<float, 2, 2>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
    } else {
      RUNTIME_ERROR("the shape is error");
    }
  } else if (3 == ndims) {
    if (1 == reduce_count) {
      mean_impl<float, 3, 1>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
    } else if (2 == reduce_count) {
      mean_impl<float, 3, 2>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
    } else if (3 == reduce_count) {
      mean_impl<float, 3, 3>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
    } else {
      RUNTIME_ERROR("the shape is error");
    }
  } else if (4 == ndims) {
    if (1 == reduce_count) {
      mean_impl<float, 4, 1>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
    } else if (2 == reduce_count) {
      mean_impl<float, 4, 2>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
    } else if (3 == reduce_count) {
      mean_impl<float, 4, 3>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
    } else if (4 == reduce_count) {
      mean_impl<float, 4, 4>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
    } else {
      RUNTIME_ERROR("the shape is error");
    }
  } else {
    RUNTIME_ERROR("the shape is error");
  }
}


}
}