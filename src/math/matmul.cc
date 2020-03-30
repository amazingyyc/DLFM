#include "common/basic.h"
#include "math/matmul.h"

namespace dlfm::math {

template <typename T>
void matmul_impl(T *x, T *y, T *z, int64_t m, int64_t k, int64_t n, bool transpose_a, bool transpose_b) {
  auto xrow = m;
  auto xcol = k;
  auto yrow = k;
  auto ycol = n;

  if (transpose_a) {
    xrow = k;
    xcol = m;
  }

  if (transpose_b) {
    yrow = n;
    ycol = k;
  }

  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> xm(x, xrow, xcol);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ym(y, yrow, ycol);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> zm(z, m, n);

  if (transpose_a && transpose_b) {
    zm.noalias() = (ym * xm).transpose();
  } else if (transpose_a) {
    zm.noalias() = xm.transpose() * ym;
  } else if (transpose_b) {
    zm.noalias() = xm * ym.transpose();
  } else {
    zm.noalias() = xm * ym;
  }
}

void matmul(const Tensor &x, const Tensor &y, Tensor &z, bool transpose_a, bool transpose_b) {
  auto m = z.shape()[0];
  auto n = z.shape()[1];

  auto k = x.shape()[1];

  if (transpose_a) {
    k = x.shape()[0];
  }

  if (x.element_type().is<float>()) {
    matmul_impl<float>(x.data<float>(), y.data<float>(), z.data<float>(), m, k, n, transpose_a, transpose_b);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}