#include "common/basic.h"
#include "math/relu.h"

namespace dlfm {
namespace math {

inline void relu_f32_block_impl(float *x, float *y, int64_t n) {
  int64_t limit = n / 4 * 4;
  int64_t i = 0;

#if defined(__ARM_NEON__)
  float32x4_t zero = vdupq_n_f32(0);

  for (; i < limit; i += 4) {
    float32x4_t x_val = vld1q_f32(x + i);

    vst1q_f32(y + i, vmaxq_f32(x_val, zero));
  }
#endif

  for (; i < n; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

void relu_f32_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *y, int64_t n) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size  = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t real_block_size = (std::min)(block_size, n - start);

    eigen_device->enqueue_with_barrier(&barrier, &relu_f32_block_impl, x + start, y + start, real_block_size);
  }

  barrier.Wait();
}

void relu(const Tensor& x, Tensor& y) {
  if (x.element_type().is<float>()) {
    relu_f32_impl(x.eigen_device().get(), x.data<float>(), y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

//---------------------------------------------------------
// relu6
inline void relu6_f32_block_impl(float *x, float *y, int64_t n) {
  int64_t limit = n / 4 * 4;
  int64_t i = 0;

#if defined(__ARM_NEON__)
  float32x4_t zero = vdupq_n_f32(0);
  float32x4_t value6 = vdupq_n_f32(6.0);

  for (; i < limit; i += 4) {
    float32x4_t x_val = vld1q_f32(x + i);

    vst1q_f32(y + i, vminq_f32(vmaxq_f32(x_val, zero), value6));
  }
#endif

  for (; i < n; ++i) {
    y[i] = (std::min)((std::max)(x[i], 0.f), 6.f);
  }
}

void relu6_f32_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *y, int64_t n) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size  = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t real_block_size = (std::min)(block_size, n - start);

    eigen_device->enqueue_with_barrier(&barrier, &relu6_f32_block_impl, x + start, y + start, real_block_size);
  }

  barrier.Wait();
}

void relu6(const Tensor &x, Tensor &y) {
  if (x.element_type().is<float>()) {
    relu6_f32_impl(x.eigen_device().get(), x.data<float>(), y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

//------------------------------------------------------------------------------------------------------
// prelu
void prelu_f32_block_impl(float *x, float w, float *y, int64_t n) {
  int64_t limit = n / 4 * 4;
  int64_t l = 0;

#if defined(__ARM_NEON__)
  float32x4_t zero = vdupq_n_f32(0);
  float32x4_t wv = vdupq_n_f32(w);

  for (; l < limit; l += 4) {
    float32x4_t xv = vld1q_f32(x + l);
    float32x4_t yv = vmaxq_f32(xv, zero) + vmulq_f32(wv, vminq_f32(zero, xv));

    vst1q_f32(y + l, yv);
  }
#endif

  for (; l < n; ++l) {
    y[l] = x[l] > 0 ? x[l] : w * x[l];
  }
}

void prelu_f32_one_channel_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float w, float *y, int64_t n) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t real_block_size = (std::min)(block_size, n - start);

    eigen_device->enqueue_with_barrier(&barrier, &prelu_f32_block_impl, x + start, w, y + start, real_block_size);
  }

  barrier.Wait();
}

void prelu_f32_multi_channel_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  float *x,
  float *w,
  float *y,
  int64_t batch,
  int64_t channel,
  int64_t height,
  int64_t width) {

  // row = batch * channel
  // col = height * width
  auto block = [](float *x, float *w, float *y, int64_t row, int64_t col, int64_t channel, int64_t row_s, int64_t row_e) {
    int64_t limit = col / 4 * 4;

    for (int64_t r = row_s; r < row_e; ++r) {
      float *xp = x + r * col;
      float *yp = y + r * col;

      float w_val = w[r % channel];

      int64_t l = 0;

#if defined(__ARM_NEON__)
      float32x4_t zero = vdupq_n_f32(0);
      float32x4_t wv = vdupq_n_f32(w_val);

      for (; l < limit; l += 4) {
        float32x4_t xv = vld1q_f32(xp + l);
        float32x4_t yv = vmaxq_f32(xv, zero) + vmulq_f32(wv, vminq_f32(zero, xv));

        vst1q_f32(yp + l, yv);
      }
#endif

      for (; l < col; ++l) {
        yp[l] = xp[l] > 0 ? xp[l] : w_val * xp[l];
      }
    }
  };

  int64_t row = batch * channel;
  int64_t col = height * width;

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size  = (row + num_threads - 1) / num_threads;

  num_threads = (row + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t row_s = i * block_size;
    int64_t row_e = std::min<int64_t>(row_s + block_size, row);

    eigen_device->enqueue_with_barrier(
      &barrier,
      block,
      x,
      w,
      y,
      row,
      col,
      channel,
      row_s,
      row_e);
  }

  barrier.Wait();
}

void prelu(const Tensor &x, const Tensor &w, Tensor &y) {
  ARGUMENT_CHECK(1 == w.size() || (1 == w.ndims() && x.shape()[1] == w.shape()[0]),
          "weight's size must be 1 or equal to channel");

  int64_t batch   = x.shape()[0];
  int64_t channel = x.shape()[1];
  int64_t height  = x.shape()[2];
  int64_t width   = x.shape()[3];

  if (x.element_type().is<float>()) {
    if (1 == w.size()) {
      prelu_f32_one_channel_impl(x.eigen_device().get(), x.data<float>(), w.data<float>()[0], y.data<float>(), x.size());
    } else {
      prelu_f32_multi_channel_impl(
              x.eigen_device().get(),
              x.data<float>(),
              w.data<float>(),
              y.data<float>(),
              batch,
              channel,
              height,
              width);
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}


}
}