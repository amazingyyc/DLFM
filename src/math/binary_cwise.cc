#include "math/binary_cwise.h"

namespace dlfm {
namespace math {

template <size_t ndims, typename T>
void add_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, const Shape &xshape, T *y, const Shape &yshape, T *z, const Shape &zshape) {
  auto new_xshape = xshape.enlarge(ndims);
  auto new_yshape = yshape.enlarge(ndims);
  auto new_zshape = zshape.enlarge(ndims);

  Eigen::array<Eigen::Index, ndims> xbroad;
  Eigen::array<Eigen::Index, ndims> ybroad;

  for (int i = 0; i < ndims; ++i) {
    if (new_xshape[i] < new_zshape[i]) {
      xbroad[i] = new_zshape[i];
    } else {
      xbroad[i] = 1;
    }

    if (new_yshape[i] < new_zshape[i]) {
      ybroad[i] = new_zshape[i];
    } else {
      ybroad[i] = 1;
    }
  }

  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;
  Eigen::array<Eigen::Index, ndims> zdims;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = new_xshape[i];
    ydims[i] = new_yshape[i];
    zdims[i] = new_zshape[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> zvec(z, zdims);

  zvec.device(*eigen_device) = xvec.broadcast(xbroad) + yvec.broadcast(ybroad);
}

void add(const Tensor &x, const Tensor &y, Tensor &z) {
  if (x.element_type().is<float>()) {
    if (1 == x.shape().ndims()) {
      add_impl<1, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (2 == x.shape().ndims()) {
      add_impl<2, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (3 == x.shape().ndims()) {
      add_impl<3, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (4 == x.shape().ndims()) {
      add_impl<4, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

template <size_t ndims, typename T>
void sub_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, const Shape &xshape, T *y, const Shape &yshape, T *z, const Shape &zshape) {
  auto new_xshape = xshape.enlarge(ndims);
  auto new_yshape = yshape.enlarge(ndims);
  auto new_zshape = zshape.enlarge(ndims);

  Eigen::array<Eigen::Index, ndims> xbroad;
  Eigen::array<Eigen::Index, ndims> ybroad;

  for (int i = 0; i < ndims; ++i) {
    if (new_xshape[i] < new_zshape[i]) {
      xbroad[i] = new_zshape[i];
    } else {
      xbroad[i] = 1;
    }

    if (new_yshape[i] < new_zshape[i]) {
      ybroad[i] = new_zshape[i];
    } else {
      ybroad[i] = 1;
    }
  }

  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;
  Eigen::array<Eigen::Index, ndims> zdims;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = new_xshape[i];
    ydims[i] = new_yshape[i];
    zdims[i] = new_zshape[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> zvec(z, zdims);

  zvec.device(*eigen_device) = xvec.broadcast(xbroad) - yvec.broadcast(ybroad);
}

void sub(const Tensor &x, const Tensor &y, Tensor &z) {
  if (x.element_type().is<float>()) {
    if (1 == x.shape().ndims()) {
      sub_impl<1, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (2 == x.shape().ndims()) {
      sub_impl<2, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (3 == x.shape().ndims()) {
      sub_impl<3, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (4 == x.shape().ndims()) {
      sub_impl<4, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

template <size_t ndims, typename T>
void multiply_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, const Shape &xshape, T *y, const Shape &yshape, T *z, const Shape &zshape) {
  auto new_xshape = xshape.enlarge(ndims);
  auto new_yshape = yshape.enlarge(ndims);
  auto new_zshape = zshape.enlarge(ndims);

  Eigen::array<Eigen::Index, ndims> xbroad;
  Eigen::array<Eigen::Index, ndims> ybroad;

  for (int i = 0; i < ndims; ++i) {
    if (new_xshape[i] < new_zshape[i]) {
      xbroad[i] = new_zshape[i];
    } else {
      xbroad[i] = 1;
    }

    if (new_yshape[i] < new_zshape[i]) {
      ybroad[i] = new_zshape[i];
    } else {
      ybroad[i] = 1;
    }
  }

  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;
  Eigen::array<Eigen::Index, ndims> zdims;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = new_xshape[i];
    ydims[i] = new_yshape[i];
    zdims[i] = new_zshape[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> zvec(z, zdims);

  zvec.device(*eigen_device) = xvec.broadcast(xbroad) * yvec.broadcast(ybroad);
}

void multiply(const Tensor &x, const Tensor &y, Tensor &z) {
  if (x.element_type().is<float>()) {
    if (1 == x.shape().ndims()) {
      multiply_impl<1, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (2 == x.shape().ndims()) {
      multiply_impl<2, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (3 == x.shape().ndims()) {
      multiply_impl<3, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (4 == x.shape().ndims()) {
      multiply_impl<4, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

template <size_t ndims, typename T>
void divide_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, const Shape &xshape, T *y, const Shape &yshape, T *z, const Shape &zshape) {
  auto new_xshape = xshape.enlarge(ndims);
  auto new_yshape = yshape.enlarge(ndims);
  auto new_zshape = zshape.enlarge(ndims);

  Eigen::array<Eigen::Index, ndims> xbroad;
  Eigen::array<Eigen::Index, ndims> ybroad;

  for (int i = 0; i < ndims; ++i) {
    if (new_xshape[i] < new_zshape[i]) {
      xbroad[i] = new_zshape[i];
    } else {
      xbroad[i] = 1;
    }

    if (new_yshape[i] < new_zshape[i]) {
      ybroad[i] = new_zshape[i];
    } else {
      ybroad[i] = 1;
    }
  }

  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;
  Eigen::array<Eigen::Index, ndims> zdims;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = new_xshape[i];
    ydims[i] = new_yshape[i];
    zdims[i] = new_zshape[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> zvec(z, zdims);

  zvec.device(*eigen_device) = xvec.broadcast(xbroad) / yvec.broadcast(ybroad);
}

void divide(const Tensor &x, const Tensor &y, Tensor &z) {
  if (x.element_type().is<float>()) {
    if (1 == x.shape().ndims()) {
      divide_impl<1, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (2 == x.shape().ndims()) {
      divide_impl<2, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (3 == x.shape().ndims()) {
      divide_impl<3, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else if (4 == x.shape().ndims()) {
      divide_impl<4, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape());
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}
}