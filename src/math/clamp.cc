#include "math/clamp.h"

namespace dlfm {
namespace math {

void clamp_f32_block_impl(float *x, float *y, float min_value, float max_value, int64_t n) {
  int64_t idx = 0;
  int64_t limit = n / 4 * 4;

#if defined(__ARM_NEON__)
  float32x4_t minv = vdupq_n_f32(min_value);
  float32x4_t maxv = vdupq_n_f32(max_value);

  for (; idx < limit; idx += 4) {
    float32x4_t xv = vld1q_f32(x + idx);
    float32x4_t yv = vminq_f32(vmaxq_f32(minv, xv), maxv);

    vst1q_f32(y + idx, yv);
  }
#endif

  for (; idx < n; ++idx) {
    y[idx] = (std::min)((std::max)(x[idx], min_value), max_value);
  }
}

void clamp_f32_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *y, float min_value, float max_value, int64_t n) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t real_block_size = (std::min)(block_size, n - start);

    eigen_device->enqueue_with_barrier(&barrier, &clamp_f32_block_impl, x + start, y + start, min_value, max_value, real_block_size);
  }

  barrier.Wait();
}

void clamp(const Tensor &x, Tensor &y, float min, float max) {
  ARGUMENT_CHECK(x.size() == y.size(), "clamp need size same");

  if (x.element_type().is<float>()) {
    clamp_f32_impl(x.eigen_device().get(), x.data<float>(), y.data<float>(), min, max, x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}


}
}