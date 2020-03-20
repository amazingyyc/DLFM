#include "math/pad.h"

namespace dlfm {
namespace math {

template <size_t ndims, typename T>
void pad_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, const Shape &xshape, T *y, const Shape &yshape, const std::vector<size_t> &paddings) {
  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = xshape[i];
    ydims[i] = yshape[i];
  }

  Eigen::array<std::pair<size_t, size_t>, ndims> e_paddings;

  for (int i = 0; i < ndims; ++i) {
    e_paddings[ndims - i - 1] = std::make_pair(paddings[2 * i], paddings[2 * i + 1]);
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);

  yvec.device(*eigen_device) = xvec.pad(e_paddings);
}

void pad(const Tensor &x, Tensor &y, std::vector<size_t> paddings) {
  while (paddings.size() < x.shape().ndims() * 2) {
    paddings.emplace_back(0);
  }

  if (x.element_type().is<float>()) {
    if (1 == x.shape().ndims()) {
      pad_impl<1, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), paddings);
    } else if (2 == x.shape().ndims()) {
      pad_impl<2, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), paddings);
    } else if (3 == x.shape().ndims()) {
      pad_impl<3, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), paddings);
    } else if (4 == x.shape().ndims()) {
      pad_impl<4, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), paddings);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}
}