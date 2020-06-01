#include "math/img_mask.h"

namespace dlfm::math {

template <typename T>
void img_mask_block_impl(T *x, uint8_t *mask, T *val, T *y, int64_t channel, int64_t height, int64_t width, int64_t start, int64_t end) {
  for (int64_t h = start; h < end; ++h) {
    for (int64_t w = 0; w < width; ++w) {
      if (1 == mask[h * width + w]) {
        for (int64_t c = 0; c < channel; ++c) {
          y[c * height * width + h * width + w] = x[c * height * width + h * width + w];
        }
      } else {
        for (int64_t c = 0; c < channel; ++c) {
          y[c * height * width + h * width + w] = val[c];
        }
      }
    }
  }
}

template <typename T>
void img_mask_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, uint8_t *mask, T *val, T *y, int64_t channel, int64_t height, int64_t width) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (height + num_threads - 1) / num_threads;

  num_threads = (height + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t end = (std::min)(start + block_size, height);

    eigen_device->enqueue_with_barrier(&barrier, &img_mask_block_impl<T>, x, mask, val, y, channel, height, width, start, end);
  }

  barrier.Wait();
}

void img_mask(const Tensor &x, const Tensor &mask, const Tensor &val, Tensor &y) {
  int64_t channel = x.shape()[0];
  int64_t height  = x.shape()[1];
  int64_t width   = x.shape()[2];

  if (x.element_type().is<float>()) {
    img_mask_impl<float>(x.eigen_device().get(), x.data<float>(), mask.data<uint8_t>(), val.data<float>(), y.data<float>(), channel, height, width);
  } else if (x.element_type().is<uint8_t>()) {
    img_mask_impl<uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), mask.data<uint8_t>(), val.data<uint8_t>(), y.data<uint8_t>(), channel, height, width);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

}