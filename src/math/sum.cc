#include "math/sum.h"

namespace dlfm::math {

void sum_f32_1d_horizontal_impl(float *x, float *y, int64_t n) {
  int64_t idx = 0;
  int64_t limit = n / 4 * 4;

  float sum = 0;

#if defined(__ARM_NEON__)
  float32x4_t sum_v = vdupq_n_f32(0);

  for (; idx < limit; idx += 4) {
    float32x4_t x_v = vld1q_f32(x + idx);
    sum_v = vaddq_f32(sum_v, x_v);
  }

  sum += sum_v[0] + sum_v[1] + sum_v[2] + sum_v[3];
#endif

  for (; idx < n; ++idx) {
    sum += x[idx];
  }

  y[0] = sum;
}

void sum_f32_1d_vertical_impl(float *x, float *y, int64_t n, int64_t stride) {
  float sum = 0;

  for (int64_t idx = 0; idx < n; ++idx) {
    sum += x[0];

    x += stride;
  }

  y[0] = sum;
}

void sum_f32_2d_horizontal_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *y, int64_t row, int64_t col) {
  Eigen::Barrier barrier((unsigned int)(row));

  for (int64_t r = 0; r < row; ++r) {
    eigen_device->enqueue_with_barrier(&barrier, &sum_f32_1d_horizontal_impl, x + r * col, y + r, col);
  }

  barrier.Wait();
}

void sum_f32_2d_vertical_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *y, int64_t row, int64_t col) {
  Eigen::Barrier barrier((unsigned int)(col));

  for (int64_t c = 0; c < col; ++c) {
    eigen_device->enqueue_with_barrier(&barrier, &sum_f32_1d_vertical_impl, x + c, y + c, row, col);
  }

  barrier.Wait();
}

template <typename T, int ndims, int reduce_count>
void sum_common_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, const Shape &xshape, T *y, const Shape &yshape) {
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

  yvec.device(*eigen_device) = xvec.sum(reduce_dims).reshape(ydims);
}

void sum(const Tensor &x, Tensor &y) {
  ARGUMENT_CHECK(x.element_type().is<float>(), "only support float");
  ARGUMENT_CHECK(x.shape().rank() == y.shape().rank(), "mean need shape rank is same");

  int ndims = x.shape().rank();

  std::vector<bool> is_reduce(ndims);

  for (int64_t i = 0; i < ndims; ++i) {
    is_reduce[i] = (x.shape()[i] != y.shape()[i]);
  }

  std::vector<bool> squeeze_reduce;
  squeeze_reduce.emplace_back(is_reduce[0]);

  for (int64_t i = 1; i < ndims; ++i) {
    if (is_reduce[i] != squeeze_reduce.back()) {
      squeeze_reduce.emplace_back(is_reduce[i]);
    }
  }

  if (1 == squeeze_reduce.size()) {
    ARGUMENT_CHECK(squeeze_reduce[0], "sum input error");

    sum_f32_1d_horizontal_impl(x.data<float>(), y.data<float>(), x.size());
  } else if (2 == squeeze_reduce.size()) {
    int64_t row = x.shape()[0];
    int64_t col = 1;

    for (int64_t i = 1; i < ndims && is_reduce[i] == is_reduce[0]; ++i) {
      row *= x.shape()[i];
    }

    col = x.shape().size() / row;

    if (is_reduce[0]) {
      sum_f32_2d_vertical_impl(x.eigen_device().get(), x.data<float>(), y.data<float>(), row, col);
    } else {
      sum_f32_2d_horizontal_impl(x.eigen_device().get(), x.data<float>(), y.data<float>(), row, col);
    }
  } else {
    int reduce_count = 0;

    for (int i = 0; i < ndims; ++i) {
      if (x.shape()[i] != y.shape()[i]) {
        reduce_count += 1;
      }

      ARGUMENT_CHECK(x.shape()[i] == y.shape()[i] || 1 == y.shape()[i], "shape error");
    }

    if (3 == ndims) {
      if (1 == reduce_count) {
        sum_common_impl<float, 3, 1>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
      } else if (2 == reduce_count) {
        sum_common_impl<float, 3, 2>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
      } else if (3 == reduce_count) {
        sum_common_impl<float, 3, 3>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
      } else {
        RUNTIME_ERROR("the shape is error");
      }
    } else if (4 == ndims) {
      if (1 == reduce_count) {
        sum_common_impl<float, 4, 1>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
      } else if (2 == reduce_count) {
        sum_common_impl<float, 4, 2>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
      } else if (3 == reduce_count) {
        sum_common_impl<float, 4, 3>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
      } else if (4 == reduce_count) {
        sum_common_impl<float, 4, 4>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
      } else {
        RUNTIME_ERROR("the shape is error");
      }
    } else {
      RUNTIME_ERROR("the shape is error");
    }
  }
}

}