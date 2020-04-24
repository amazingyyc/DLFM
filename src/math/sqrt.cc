#include "math/sqrt.h"

namespace dlfm {
namespace math {

#if defined(__ARM_NEON__)
float32x4_t vsqrtq_f32(float32x4_t q_x) {
  const float32x4_t q_step_0 = vrsqrteq_f32(q_x);
  // step
  const float32x4_t q_step_parm0 = vmulq_f32(q_x, q_step_0);
  const float32x4_t q_step_result0 = vrsqrtsq_f32(q_step_parm0, q_step_0);
  // step
  const float32x4_t q_step_1 = vmulq_f32(q_step_0, q_step_result0);
  const float32x4_t q_step_parm1 = vmulq_f32(q_x, q_step_1);
  const float32x4_t q_step_result1 = vrsqrtsq_f32(q_step_parm1, q_step_1);
  // take the res
  const float32x4_t q_step_2 = vmulq_f32(q_step_1, q_step_result1);
  // mul by x to get sqrt, not rsqrt
  return vmul_f32(q_x, q_step_2);
}
#endif

void sqrt_f32_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *y, int64_t n) {
  auto block = [](float *x, float *y, int64_t n) {
    int64_t idx = 0;
    int64_t limit = n / 4 * 4;

#if defined(__ARM_NEON__)
    for (; idx < limit; idx += 4) {
      float32x4_t yv = vsqrtq_f32(vld1q_f32(x + idx));

      vst1q_f32(y + idx, yv);
    }
#endif

    for (; idx < n; ++idx) {
      y[idx] = std::sqrt(x[idx]);
    }
  };

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t real_block_size = (std::min)(block_size, n - start);

    eigen_device->enqueue_with_barrier(&barrier, block, x + start, y + start, real_block_size);
  }

  barrier.Wait();
}

void sqrt(const Tensor &x, Tensor &y) {
  if (x.element_type().is<float>()) {
    sqrt_f32_impl(x.eigen_device().get(), x.data<float>(), y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

}
}