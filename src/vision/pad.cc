#include "vision/pad.h"

namespace dlfm::vision {

template <typename T>
void pad_block_impl(
  std::shared_ptr<Device> device,
  T *x,
  T *y,
  T *value
  int64_t input_height,
  int64_t input_width,
  int64_t output_height,
  int64_t output_width,
  int64_t channel,
  int64_t top,
  int64_t bottom,
  int64_t left,
  int64_t right,
  int64_t start_index,
  int64_t end_index) {
  int64_t channel_bytes = sizeof(T) * channel;

  for (int64_t idx = start_index; idx < end_index; ++idx) {
    if (idx < top || idx >= (top + input_height)) {
      // Top/bottom pad
      T *y_offset = y + idx * output_height * channel;
      for (int64_t j = 0; j < output_width; ++j) {
        device->memcpy(y_offset, value, channel_bytes);

        y_offset += channel;
      }
    } else {
      // left
      T *y_offset = y + idx * output_height * channel;
      T *x_offset = x + (idx - top) * input_width * channel;

      for (int64_t j = 0; j < left; ++j) {
        device->memcpy(y_offset, value, channel_bytes);

        y_offset += channel;
      }

      // mid
      device->memcpy(y_offset, x_offset, sizeof(T) * input_width * channel);

      y_offset += input_width * channel;

      // right
      for (int64_t j = left + input_width; j < output_width; ++j) {
        device->memcpy(y_offset, value, channel_bytes);

        y_offset += channel;
      }
    }
  }
}

template <typename T>
void pad_impl(
  std::shared_ptr<Device> device,
  T *x,
  T *y,
  T *value
  int64_t input_height,
  int64_t input_width,
  int64_t output_height,
  int64_t output_width,
  int64_t channel,
  int64_t top,
  int64_t bottom,
  int64_t left,
  int64_t right) {
  int64_t num_threads = (int64_t)device->eigen_device()->numThreads();
  int64_t block_size = (output_height + num_threads - 1) / num_threads;
  int64_t need_num_threads = (output_height + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_index = i * block_size;
    int64_t end_index = std::min<int64_t>(start_index + block_size, output_height);

    device->eigen_device()->enqueue_with_barrier(
      &barrier,
      &pad_block_impl<T>,
      x,
      y,
      value,
      input_height,
      input_width,
      output_height,
      output_width,
      channel,
      top,
      bottom,
      left,
      right,
      start_index,
      end_index);
  }

  barrier.Wait();
}

Tensor pad(const Tensor &x, const Tensor &value, int64_t top, int64_t bottom, int64_t left, int64_t right) {
  ARGUMENT_CHECK(3 == x.shape().ndims() && 1 == value.shape().ndims(), "x shape dimension must be 3, value must be 1.");
  ARGUMENT_CHECK(x.shape()[2] == value.shape()[0], "x/value laste diemension must same.");
  ARGUMENT_CHECK(top >= 0 && bottom >= 0 && left >= 0 && right >= 0, "top/bottom/left/right must >= 0.");
  ARGUMENT_CHECK(x.element_type() == value.element_type(), "x/value element type must be same.");

  int64_t input_height = x.shape()[0];
  int64_t input_width = x.shape()[1];
  int64_t channel = x.shape()[2];

  int64_t output_height = top + input_height + bottom;
  int64_t output_width = left + input_width + right;

  auto y = Tensor::create({output_height, output_width, channel}, x.element_type());

  if (x.element_type().is<float>()) {
    pad_impl(
      x.device(),
      x.data<float>(),
      y.data<float>(),
      value.data<float>(),
      input_height,
      input_width,
      output_height,
      output_width,
      channel,
      top,
      bottom,
      left,
      right);
  } else if (x.element_type().is<uint8_t>()) {
    pad_impl(
      x.device(),
      x.data<uint8_t>(),
      y.data<uint8_t>(),
      value.data<uint8_t>(),
      input_height,
      input_width,
      output_height,
      output_width,
      channel,
      top,
      bottom,
      left,
      right);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

Tensor pad(const Tensor &x, const Tensor &value, std::vector<int64_t> paddings) {
  ARGUMENT_CHECK(4 == paddings.size(), "paddings size must be 4");

  return pad(x, value, padding[0], padding[1], padding[2], padding[3]);
}

}