#include "common/basic.h"
#include "math/norm2d.h"

namespace dlfm::math {

void norm2d_f32_block_impl(
  float *x,
  float *y,
  float eps,
  int64_t row,
  int64_t col,
  int64_t idx) {
  int64_t col_limit = col / 4 * 4;

  float mean = 0;
  float var  = 0;

  float* xptr = x + idx * col;
  float* yptr = y + idx * col;

  // cal mean
  {
    double sum = 0;
    int64_t c = 0;

#if defined(__ARM_NEON__)
    // calculate mean
    float32x4_t sumv = vdupq_n_f32(0);

    for (; c < col_limit; c += 4) {
      sumv = vaddq_f32(sumv, vld1q_f32(xptr + c));
    }

    sum += sumv[0] + sumv[1] + sumv[2] + sumv[3];
#endif

    for (; c < col; ++c) {
      sum += (double)(xptr[c]);
    }

    mean = (float)(sum / double(col));
  }

  {
    double sum = 0;
    int64_t c = 0;

#if defined(__ARM_NEON__)
    // calcuate variance
    float32x4_t meanv = vdupq_n_f32(mean);
    float32x4_t sumv = vdupq_n_f32(0);

    for (; c < col_limit; c += 4) {
      float32x4_t diff = vsubq_f32(vld1q_f32(xptr + c), meanv);

      sumv = vaddq_f32(sumv, vmulq_f32(diff, diff));
    }

    sum += sumv[0] + sumv[1] + sumv[2] + sumv[3];
#endif

    for (; c < col; ++c) {
      double diff = xptr[c] - mean;
      sum += diff * diff;
    }

    var = (float)(sum / double(col));
  }

  float var_rsqrt = 1.0 / std::sqrt(var + eps);

  // calculate y
  {
    int64_t c = 0;

#if defined(__ARM_NEON__)
    float32x4_t meanv = vdupq_n_f32(mean);
    float32x4_t var_rsqrtv = vdupq_n_f32(var_rsqrt);

    for (; c < col_limit; c += 4) {
      float32x4_t xv = vld1q_f32(xptr + c);
      float32x4_t yv = vmulq_f32(vsubq_f32(xv, meanv), var_rsqrtv);

      vst1q_f32(yptr + c, yv);
    }
#endif

    for (; c < col; ++c) {
      yptr[c] = (xptr[c] - mean) * var_rsqrt;
    }
  }
}

void norm2d_f32_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  float *x,
  float *y,
  float eps,
  int64_t row,
  int64_t col) {
  Eigen::Barrier barrier((unsigned int)(row));

  for (int64_t i = 0; i < row; ++i) {
    eigen_device->enqueue_with_barrier(&barrier, &norm2d_f32_block_impl, x, y, eps, row, col);
  }

  barrier.Wait();
}

void norm2d(const Tensor &x, Tensor &y, float eps) {
  ARGUMENT_CHECK(x.shape() == y.shape(), "x/y shape must same");
  ARGUMENT_CHECK(2 == x.shape().rank(), "rank must be 2");

  int64_t row = x.shape()[0];
  int64_t col = x.shape()[1];

  if (x.element_type().is<float>()) {
    norm2d_f32_impl(
      x.eigen_device().get(),
      x.data<float>(),
      y.data<float>(),
      eps,
      row,
      col);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}