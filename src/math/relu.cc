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
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

}
}