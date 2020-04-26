#include "math/cast.h"

namespace dlfm {
namespace math {

template <typename From, typename To>
void cast_block_impl(From *x, To *y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    y[i] = static_cast<To>(x[i]);
  }
}

template <typename From, typename To>
void cast_impl(Eigen::ThreadPoolDevice *eigen_device, From *x, To *y, int64_t n) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t real_block_size = (std::min)(block_size, n - start);

    eigen_device->enqueue_with_barrier(&barrier, &cast_block_impl<From, To>, x + start, y + start, real_block_size);
  }

  barrier.Wait();
}

void cast(const Tensor &x, Tensor &y) {
  auto xtype = x.element_type();
  auto ytype = y.element_type();

  if (xtype.is<float>() && ytype.is<uint8_t>()) {
    cast_impl<float, uint8_t>(x.eigen_device().get(), x.data<float>(), y.data<uint8_t>(), x.size());
  } else if (xtype.is<uint8_t>() && ytype.is<float>()) {
    cast_impl<uint8_t, float>(x.eigen_device().get(), x.data<uint8_t>(), y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("cast from:" << xtype.name() << " to:" << ytype.name() << " not support.");
  }
}

}
}