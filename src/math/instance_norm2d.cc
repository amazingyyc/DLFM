#include "common/basic.h"
#include "math/instance_norm2d.h"

namespace dlfm::math {

// x [b, c, h, w] -> [b * c, h * w] -> [row, col]
// scale, shift [c]
// mean variance [b, c]
// y [b, c, h, w]
void instance_norm2d_f32_block_impl(
  float *x,
  float eps,
  float *y,
  int64_t b,
  int64_t c,
  int64_t h,
  int64_t w,
  int64_t idx) {
  int64_t row = b * c;
  int64_t col = h * w;

  int64_t col_limit = col / 4 * 4;

  float mean = 0;
  float variance = 0;

  float* xptr = x + idx * col;
  float* yptr = y + idx * col;

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

    variance = (float)(sum / double(col));
  }

  float variance_rsqrt = 1.0 / std::sqrt(variance + eps);

  // calculate y
  {
    int64_t c = 0;

#if defined(__ARM_NEON__)
    float32x4_t meanv = vdupq_n_f32(mean);
    float32x4_t variance_rsqrtv = vdupq_n_f32(variance_rsqrt);

    for (; c < col_limit; c += 4) {
      float32x4_t xv = vld1q_f32(xptr + c);
      float32x4_t yv = vmulq_f32(vsubq_f32(xv, meanv), variance_rsqrtv);

      vst1q_f32(yptr + c, yv);
    }
#endif

    for (; c < col; ++c) {
      yptr[c] = (xptr[c] - mean) * variance_rsqrt;
    }
  }
}

void instance_norm2d_f32_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  float *x,
  float eps,
  float *y,
  int64_t b,
  int64_t c,
  int64_t h,
  int64_t w) {
  Eigen::Barrier barrier((unsigned int)(b * c));

  for (int64_t i = 0; i < b * c; ++i) {
    eigen_device->enqueue_with_barrier(&barrier, &instance_norm2d_f32_block_impl, x, eps, y, b, c, h, w, i);
  }

  barrier.Wait();
}
// instance norm2d need input is [b, c, h, w],
void instance_norm2d(const Tensor &input, float eps, Tensor &output) {
  int64_t b = input.shape()[0];
  int64_t c = input.shape()[1];
  int64_t h = input.shape()[2];
  int64_t w = input.shape()[3];

  if (input.element_type().is<float>()) {
    instance_norm2d_f32_impl(
      input.eigen_device().get(),
      input.data<float>(),
      eps,
      output.data<float>(),
      b,
      c,
      h,
      w);
  } else {
    RUNTIME_ERROR("element type:" << input.element_type().name() << " nor support!");
  }
}

//------------------------------------------------------------------------------------------------------

// x [b, c, h, w] -> [b * c, h * w] -> [row, col]
// scale, shift [c]
// mean variance [b, c]
// y [b, c, h, w]
void instance_norm2d_float_block_impl(
  float *x,
  float *scale,
  float *shift,
  float eps,
  float *y,
  int64_t b,
  int64_t c,
  int64_t h,
  int64_t w,
  int64_t idx) {
  int64_t row = b * c;
  int64_t col = h * w;

  int64_t col_limit = col / 4 * 4;

  float mean = 0;
  float variance = 0;

  float* xptr = x + idx * col;
  float* scaleptr = scale + (idx % c);
  float* shiftptr = shift + (idx % c);
  float* yptr = y + idx * col;

  {
    int64_t c = 0;

#if defined(__ARM_NEON__)
    // calculate mean
    float32x4_t sumv = vdupq_n_f32(0);

    for (; c < col_limit; c += 4) {
      sumv = vaddq_f32(sumv, vld1q_f32(xptr + c));
    }

    //float32_t vgetq_lane_f32(float32x4_t vec, __constrange(0,3) int lane);
    mean += sumv[0] + sumv[1] + sumv[2] + sumv[3];
#endif

    for (; c < col; ++c) {
      mean += xptr[c];
    }

    mean /= float(col);
  }

  {
    int64_t c = 0;

#if defined(__ARM_NEON__)
    // calcuate variance
    float32x4_t meanv = vdupq_n_f32(mean);
    float32x4_t sumv = vdupq_n_f32(0);

    for (; c < col_limit; c += 4) {
      float32x4_t diff = vsubq_f32(vld1q_f32(xptr + c), meanv);

      sumv = vaddq_f32(sumv, vmulq_f32(diff, diff));
    }

    variance += sumv[0] + sumv[1] + sumv[2] + sumv[3];
#endif

    for (; c < col; ++c) {
      float diff = xptr[c] - mean;
      variance += diff * diff;
    }

    variance /= float(col);
  }

  float variance_rsqrt = 1.0 / std::sqrt(variance + eps);

  // calculate y
  {
    int64_t c = 0;

#if defined(__ARM_NEON__)
    float32x4_t meanv = vdupq_n_f32(mean);
    float32x4_t variance_rsqrtv = vdupq_n_f32(variance_rsqrt);

    float32x4_t scalev = vdupq_n_f32(scaleptr[0]);
    float32x4_t shiftv = vdupq_n_f32(shiftptr[0]);

    for (; c < col_limit; c += 4) {
      float32x4_t xv = vld1q_f32(xptr + c);
      float32x4_t tv = vmulq_f32(vsubq_f32(xv, meanv), variance_rsqrtv);
      float32x4_t yv = vaddq_f32(vmulq_f32(tv, scalev), shiftv);

      vst1q_f32(yptr + c, yv);
    }
#endif

    for (; c < col; ++c) {
      yptr[c] = (xptr[c] - mean) * variance_rsqrt * scaleptr[0] + shiftptr[0];
    }
  }
}

void instance_norm2d_float_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  float *x,
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
    eigen_device->enqueue_with_barrier(&barrier, &instance_norm2d_float_block_impl, x, scale, shift, eps, y, b, c, h, w, i);
  }

  barrier.Wait();
}

void instance_norm2d(const Tensor &input, const Tensor &scale, const Tensor &shift, float eps, Tensor &output) {
  int64_t b = input.shape()[0];
  int64_t c = input.shape()[1];
  int64_t h = input.shape()[2];
  int64_t w = input.shape()[3];

  if (input.element_type().is<float>()) {
    instance_norm2d_float_impl(
      input.eigen_device().get(),
      input.data<float>(),
      scale.data<float>(),
      shift.data<float>(),
      eps,
      output.data<float>(),
      b,
      c,
      h,
      w);
  } else {
    RUNTIME_ERROR("element type:" << input.element_type().name() << " nor support!");
  }
}

}