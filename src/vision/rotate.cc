#include "vision/rotate.h"

namespace dlfm::vision {

template <typename T>
void rotate_right_90_block_impl(
  std::shared_ptr<Device> device,
  T *x,
  T *y,
  int64_t x_height,
  int64_t x_width,
  int64_t channel,
  int64_t start,
  int64_t end) {
  int64_t bytes_count = sizeof(T) * channel;

  for (int64_t i = start; i < end; ++i) {
    for (int64_t j = 0; j < x_height; ++j) {
      device->memcpy(y + i * x_height * channel + j * channel, x + j * x_width * channel + i * channel, bytes_count);
    }
  }
}

template <typename T>
void rotate_right_90_impl(
  std::shared_ptr<Device> device,
  T *x,
  T *y,
  int64_t x_height,
  int64_t x_width,
  int64_t channel) {
  int64_t num_threads = (int64_t)device->eigen_device()->numThreads();
  int64_t block_size = (x_width + num_threads - 1) / num_threads;
  int64_t need_num_threads = (x_width + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_index = i * block_size;
    int64_t end_index = std::min<int64_t>(start_index + block_size, x_width);

    device->eigen_device()->enqueue_with_barrier(
      &barrier,
      &rotate_right_90_block_impl<T>,
      device,
      x,
      y,
      x_height,
      x_width,
      channel,
      start_index,
      end_index);
  }

  barrier.Wait();
}

// rotate the image right_90 degree
Tensor rotate_right_90(const Tensor &x) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "x ndims must be 3.");

  int64_t x_height = x.shape()[0];
  int64_t x_width = x.shape()[1];
  int64_t channel = x.shape()[2];

  auto y = Tensor::create({ x_width , x_height , channel }, x.element_type());

  if (x.element_type().is<float>()) {
    rotate_right_90_impl(
      x.device(),
      x.data<float>(),
      y.data<float>(),
      x_height,
      x_width,
      channel
    );
  } else if (x.element_type().is<uint8_t>()) {
    rotate_right_90_impl(
      x.device(),
      x.data<uint8_t>(),
      y.data<uint8_t>(),
      x_height,
      x_width,
      channel
    );
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}