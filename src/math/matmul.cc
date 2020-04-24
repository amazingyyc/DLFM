#include "common/basic.h"
#include "math/matmul.h"

namespace dlfm::math {

template <typename T>
void matmul_impl(
  T *x,
  int64_t xrow,
  int64_t xcol,
  T *y,
  int64_t yrow,
  int64_t ycol,
  T *z,
  int64_t zrow,
  int64_t zcol,
  bool transpose_a,
  bool transpose_b) {
  // ios compiler not support openmp, matmul is single thread
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> xm(x, xrow, xcol);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ym(y, yrow, ycol);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> zm(z, zrow, zcol);

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
  if (x.element_type().is<float>()) {
    matmul_impl<float>(
      x.data<float>(),
      x.shape()[0],
      x.shape()[1],
      y.data<float>(),
      y.shape()[0],
      y.shape()[1],
      z.data<float>(),
      z.shape()[0],
      z.shape()[1],
      transpose_a,
      transpose_b);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}