#include "vision/crop.h"

namespace dlfm::vision {

template <typename T>
void crop_block_impl(
  std::shared_ptr<Device> device,
  T *x,
  T *y,
  int64_t input_height,
  int64_t input_width,
  int64_t offset_y,
  int64_t offset_x,
  int64_t output_height,
  int64_t output_width,
  int64_t channel,
  int64_t start_idx,
  int64_t end_idx) {
  size_t byte_count = sizeof(T) * output_width * channel;

  for (int64_t oy = start_idx; oy < end_idx; ++oy) {
    T *x_ptr = x + (offset_y + oy) * input_width * channel + offset_x * channel;
    T *y_ptr = y + oy * output_width * channel;

    device->memcpy(y_ptr, x_ptr, byte_count);
  }
}

template <typename T>
void crop_impl(
  std::shared_ptr<Device> device,
  T *x,
  T *y,
  int64_t input_height,
  int64_t input_width,
  int64_t offset_y,
  int64_t offset_x,
  int64_t output_height,
  int64_t output_width,
  int64_t channel) {
  int64_t num_threads = (int64_t)device->eigen_device()->numThreads();
  int64_t block_size = (output_height + num_threads - 1) / num_threads;
  int64_t need_num_threads = (output_height + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_index = i * block_size;
    int64_t end_index = std::min<int64_t>(start_index + block_size, output_height);

    device->eigen_device()->enqueue_with_barrier(
      &barrier,
      &crop_block_impl<T>,
      device,
      x,
      y,
      input_height,
      input_width,
      offset_y,
      offset_x,
      output_height,
      output_width,
      channel,
      start_index,
      end_index);
  }

  barrier.Wait();
}

Tensor crop(const Tensor &x, std::vector<int64_t> offset, std::vector<int64_t> size) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "crop need x dimension is 3");
  ARGUMENT_CHECK(2 == offset.size() && 2 == size.size(), "offset/size must 2 dimension");

  int64_t input_height = x.shape()[0];
  int64_t input_width = x.shape()[1];
  int64_t channel = x.shape()[2];

  int64_t offset_y = offset[0];
  int64_t offset_x = offset[1];

  int64_t output_height = size[0];
  int64_t output_width = size[1];

  ARGUMENT_CHECK(0 <= offset_y
                 && offset_y < input_height
                 && 0 <= offset_x
                 && offset_x < input_width
                 && output_height > 0
                 && output_width > 0
                 && offset_y + output_height <= input_height
                 && offset_x + output_width <= input_width,
                 "shape error, offset_y:" << offset_y
                 << ",offset_x:" << offset_x
                 << ",output_height:" << output_height
                 << ",output_width" << output_width);

  auto y = Tensor::create({ output_height , output_width , channel }, x.element_type());

  if (x.element_type().is<float>()) {
    crop_impl(
      x.device(),
      x.data<float>(),
      y.data<float>(),
      input_height,
      input_width,
      offset_y,
      offset_x,
      output_height,
      output_width,
      channel);
  } else if (x.element_type().is<uint8_t>()) {
    crop_impl(
      x.device(),
      x.data<uint8_t>(),
      y.data<uint8_t>(),
      input_height,
      input_width,
      offset_y,
      offset_x,
      output_height,
      output_width,
      channel);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }

  return y;
}

}