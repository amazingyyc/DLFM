#include "common/basic.h"
#include "math/batch_norm2d.h"

namespace dlfm::math {

void batch_norm2d_f32_block_impl(
  float *x,
  float *mean,
  float *variance,
  float *scale,
  float *shift,
  float eps,
  float *y,
  int64_t b,
  int64_t c,
  int64_t h,
  int64_t w,
  int64_t idx) {
  int64_t b_idx = idx / c;
  int64_t c_idx = idx % c;

  float mean_v = mean[c_idx];
  float variance_rsqrt_v = 1.0 / std::sqrt(variance[c_idx] + eps);
  float scale_v = scale[c_idx];
  float shift_v = shift[c_idx];

  float *x_ptr = x + idx * h * w;
  float *y_ptr = y + idx * h * w;

  int64_t limit = (h * w) / 4 * 4;
  int64_t l = 0;

#if defined(__ARM_NEON__)
  float32x4_t mean_v_t = vdupq_n_f32(mean_v);
  float32x4_t variance_rsqrt_v_t = vdupq_n_f32(variance_rsqrt_v);

  float32x4_t scale_v_t = vdupq_n_f32(scale_v);
  float32x4_t shift_v_t = vdupq_n_f32(shift_v);

  for (; l < limit; l + 4) {
    float32x4_t xv = vld1q_f32(x_ptr + l);
    float32x4_t tv = vmulq_f32(vsubq_f32(xv, mean_v_t), variance_rsqrt_v_t);
    float32x4_t yv = vaddq_f32(vmulq_f32(tv, scale_v_t), shift_v_t);

    vst1q_f32(y_ptr + l, yv);
  }
#endif

  for (; l < h * w; ++l) {
    y_ptr[l] = (x_ptr[l] - mean_v) * variance_rsqrt_v * scale_v + shift_v;
  }
}

void batch_norm2d_f32_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  float *x,
  float *mean,
  float *variance,
  float *scale,
  float *shift,
  float eps,
  float *y,
  int64_t b,
  int64_t c,
  int64_t h,
  int64_t w) {
  Eigen::Barrier barrier((unsigned int)(b * c));

  for (int64_t i = 0; i < b * c; ++i) {
    eigen_device->enqueue_with_barrier(
      &barrier,
      &batch_norm2d_f32_block_impl,
      x, mean, variance, scale, shift, eps, y, b, c, h, w, i);
  }

  barrier.Wait();
}

void batch_norm2d(
  const Tensor &x, // [b, c, h, w]
  const Tensor &mean, // [1, c, 1, 1]
  const Tensor &variance, // [1, c, 1, 1]
  const Tensor &scale, // [1, c, 1, 1]
  const Tensor &shift, // [1, c, 1 ,1]
  float eps,
  Tensor &y) {
  int64_t b = x.shape()[0];
  int64_t c = x.shape()[1];
  int64_t h = x.shape()[2];
  int64_t w = x.shape()[3];

  if (x.element_type().is<float>()) {
    batch_norm2d_f32_impl(
      x.eigen_device().get(),
      x.data<float>(),
      mean.data<float>(),
      variance.data<float>(),
      scale.data<float>(),
      shift.data<float>(),
      eps,
      y.data<float>(),
      b,
      c,
      h,
      w);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

}