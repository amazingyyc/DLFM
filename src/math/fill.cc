#include "math/fill.h"

namespace dlfm {
namespace math {

template <typename T>
void fill_block_impl(T *p, T value, int64_t n) {
  int64_t l = 0;
  int64_t limit = n / 4 * 4;

  for (; l < limit; l += 4) {
    p[l] = value;
    p[l + 1] = value;
    p[l + 2] = value;
    p[l + 3] = value;
  }

  for (; l < n; ++l) {
    p[l] = value;
  }
}

template <typename T>
void fill_impl(Eigen::ThreadPoolDevice *eigen_device, T *p, T value, int64_t n) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t real_block_size = (std::min)(block_size, n - start);

    eigen_device->enqueue_with_barrier(&barrier, &fill_block_impl, p + start, value, real_block_size);
  }

  barrier.Wait();
}

void fill(Tensor &tensor, float value) {
  if (tensor.element_type().is<float>()) {
    fill_impl<float>(tensor.eigen_device().get(), tensor.data<float>(), value, tensor.size());
  } else if (tensor.element_type().is<uint8_t>()) {
    fill_impl<uint8_t>(tensor.eigen_device().get(), tensor.data<uint8_t>(), (uint8_t)value, tensor.size());
  } else {
    RUNTIME_ERROR("element type:" << tensor.element_type().name() << " nor support!");
  }
}

}
}