#include "common/basic.h"
#include "math/relu.h"

namespace dlfm {
namespace math {

#if defined(__ARM_NEON__)
inline void relu_float_neon_block_impl(float *x, float *y, int64_t start, int64_t end) {
  int64_t limit = start + (end - start) / 4 * 4;
  int64_t i = start;

  float32x4_t zero = vdupq_n_f32(0);

  for (; i < limit; i += 4) {
    float32x4_t x_val = vld1q_f32(x + i);

    vst1q_f32(y + i, vmaxq_f32(x_val, zero));
  }

  for (; i < end; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

void relu_float_neon_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *y, int64_t n) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size  = (n + num_threads - 1) / num_threads;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  auto block = [&barrier](float *x, float *y, int64_t start, int64_t end) {
    relu_float_neon_block_impl(x, y, start, end);

    barrier.Notify();
  };

  for (int64_t i = 0; i < num_threads; ++i) {
    int start = i * block_size;
    int end = std::min<int64_t>(start + block_size, n);

    eigen_device->enqueueNoNotification(block, x, y, start, end);
  }

  barrier.Wait();
}
#else
template <typename T>
void relu_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec.cwiseMax(T(0));
}
#endif

void relu(const Tensor& x, Tensor& y) {
  if (x.element_type().is<float>()) {
#if defined(__ARM_NEON__)
    relu_float_neon_impl(x.eigen_device().get(), x.data<float>(), y.data<float>(), x.size());
#else
    relu_impl<float>(x.eigen_device().get(), x.data<float>(), y.data<float>(), x.size());
#endif
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

}
}