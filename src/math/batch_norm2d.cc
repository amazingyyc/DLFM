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
  int64_t batch,
  int64_t channel,
  int64_t height,
  int64_t width,
  int64_t start_channel,
  int64_t end_channel) {
  for (int64_t idx = start_channel; idx < end_channel; ++idx) {
    int64_t batch_idx = idx / channel;
    int64_t channel_idx = idx % channel;

    float mean_v = mean[channel_idx];
    float variance_rsqrt_v = 1.0 / std::sqrt(variance[channel_idx] + eps);
    float scale_v = scale[channel_idx];
    float shift_v = shift[channel_idx];

    float *x_ptr = x + idx * height * width;
    float *y_ptr = y + idx * height * width;

    int64_t limit = (height * width) / 8 * 8;
    int64_t l = 0;

#if defined(__ARM_NEON__)
    float32x4_t mean_v_t = vdupq_n_f32(mean_v);
    float32x4_t variance_rsqrt_v_t = vdupq_n_f32(variance_rsqrt_v);

    float32x4_t scale_v_t = vdupq_n_f32(scale_v);
    float32x4_t shift_v_t = vdupq_n_f32(shift_v);

    float32x4_t xv1;
    float32x4_t xv2;
    float32x4_t tv1;
    float32x4_t tv2;

    for (; l < limit; l += 8) {
      xv1 = vld1q_f32(x_ptr + l);
      xv2 = vld1q_f32(x_ptr + l + 4);

      tv1 = vmulq_f32(vsubq_f32(xv1, mean_v_t), variance_rsqrt_v_t);
      tv2 = vmulq_f32(vsubq_f32(xv2, mean_v_t), variance_rsqrt_v_t);

      vst1q_f32(y_ptr + l, vaddq_f32(vmulq_f32(tv1, scale_v_t), shift_v_t));
      vst1q_f32(y_ptr + l + 4, vaddq_f32(vmulq_f32(tv2, scale_v_t), shift_v_t));
    }
#endif

    for (; l < height * width; ++l) {
      y_ptr[l] = (x_ptr[l] - mean_v) * variance_rsqrt_v * scale_v + shift_v;
    }
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
  int64_t n = b * c;

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t end = (std::min)(start + block_size, n);

    eigen_device->enqueue_with_barrier(
      &barrier,
      &batch_norm2d_f32_block_impl,
      x, mean, variance, scale, shift, eps, y, b, c, h, w, start, end);
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