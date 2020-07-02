#include "vision/convert.h"

namespace dlfm::vision {

template <typename T>
void convert_block_impl(
  T *x,
  T *y,
  int64_t height,
  int64_t width,
  int64_t x_channel,
  int64_t y_channel,
  const std::vector<size_t> &idxs,
  int64_t start,
  int64_t end) {
  for (int64_t h = start; h < end; ++h) {
    for (int64_t w = 0; w < width; ++w) {
      T *x_offset = x + h * width * x_channel + w * x_channel;
      T *y_offset = y + h * width * y_channel + w * y_channel;

      for (int64_t c = 0; c < y_channel; ++c) {
        y_offset[c] = x_offset[idxs[c]];
      }
    }
  }
}

template <typename T>
void convert_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  T *x,
  T *y,
  int64_t height,
  int64_t width,
  int64_t x_channel,
  int64_t y_channel,
  const std::vector<size_t> &idxs) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (height + num_threads - 1) / num_threads;
  int64_t need_num_threads = (height + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_index = i * block_size;
    int64_t end_index = std::min<int64_t>(start_index + block_size, height);

    eigen_device->enqueue_with_barrier(
      &barrier,
      &convert_block_impl<T>,
      x,
      y,
      height,
      width,
      x_channel,
      y_channel,
      idxs,
      start_index,
      end_index);
  }

  barrier.Wait();
}

Tensor convert(const Tensor &x, std::vector<size_t> idx) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "convert need input dimension is 3.");

  int64_t height = x.shape()[0];
  int64_t width  = x.shape()[1];

  int64_t x_channel = x.shape()[2];
  int64_t y_channel = idx.size();

  for (auto i : idx) {
    ARGUMENT_CHECK(i >= 0 && i < x_channel, "idx out of range");
  }

  auto y = Tensor::create({height, width, y_channel}, x.element_type());

  if (x.element_type().is<float>()) {
    convert_impl(
      x.eigen_device().get(),
      x.data<float>(),
      y.data<float>(),
      height,
      width,
      x_channel,
      y_channel,
      idx);
  } else if (x.element_type().is<uint8_t>()) {
    convert_impl(
      x.eigen_device().get(),
      x.data<uint8_t>(),
      y.data<uint8_t>(),
      height,
      width,
      x_channel,
      y_channel,
      idx);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }

  return y;
}

Tensor bgra_2_rgb(const Tensor &x) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "x ndims must be 3");
  ARGUMENT_CHECK(4 == x.shape()[2], "last dimension must be 4");

  return convert(x, {2, 1, 0});
}

Tensor bgra_2_rgba(const Tensor &x) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "x ndims must be 3");
  ARGUMENT_CHECK(4 == x.shape()[2], "last dimension must be 4");

  return convert(x, { 2, 1, 0, 3 });
}

}