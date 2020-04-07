#include "math/slice.h"

namespace dlfm::math {

template <size_t ndims, typename T>
void slice_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  T *x,
  const Shape &xshape,
  T *y,
  const Shape &yshape,
  const std::vector<int64_t> &off,
  const std::vector<int64_t> &ext) {
  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;
  Eigen::array<Eigen::Index, ndims> offsets;
  Eigen::array<Eigen::Index, ndims> extents;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = xshape[i];
    ydims[i] = yshape[i];

    offsets[i] = off[i];
    extents[i] = ext[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);

  yvec.device(*eigen_device) = xvec.slice(offsets, extents);
}

void slice(const Tensor &x, Tensor &y, const std::vector<int64_t> &offsets, const std::vector<int64_t> &extents) {
  if (x.element_type().is<float>()) {
    if (1 == x.shape().ndims()) {
      slice_impl<1, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), offsets, extents);
    } else if (2 == x.shape().ndims()) {
      slice_impl<2, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), offsets, extents);
    } else if (3 == x.shape().ndims()) {
      slice_impl<3, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), offsets, extents);
    } else if (4 == x.shape().ndims()) {
      slice_impl<4, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), offsets, extents);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else if (x.element_type().is<uint8_t >()) {
    if (1 == x.shape().ndims()) {
      slice_impl<1, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), offsets, extents);
    } else if (2 == x.shape().ndims()) {
      slice_impl<2, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), offsets, extents);
    } else if (3 == x.shape().ndims()) {
      slice_impl<3, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), offsets, extents);
    } else if (4 == x.shape().ndims()) {
      slice_impl<4, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), offsets, extents);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}