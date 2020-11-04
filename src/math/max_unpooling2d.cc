#include "common/basic.h"
#include "math/max_unpooling2d.h"

namespace dlfm::math {

template <typename T>
void max_unpooling2d_block_impl(
  T *input,
  int64_t *indices,
  T *output,
  int64_t start_channel,
  int64_t end_channel,
  int64_t input_height,
  int64_t input_width,
  int64_t output_height,
  int64_t output_width) {
  for (int64_t c = start_channel; c < end_channel; ++c) {
    for (int64_t i_idx = 0; i_idx < input_height * input_width; ++i_idx) {
      int64_t o_idx = indices[c * input_height * input_width + i_idx];

      if (o_idx >= 0 && o_idx < output_height * output_width) {
        output[c * output_height * output_width + o_idx] = input[c * input_height * input_width + i_idx];
      }
    }
  }
}

template <typename T>
void max_unpooling2d_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  T *x,
  const Shape &x_shape,
  int64_t *indices,
  const Shape &indices_shape,
  T *y,
  const Shape &y_shape) {
  int64_t channel = x_shape[0] * x_shape[1];
  int64_t input_height = x_shape[2];
  int64_t input_width = x_shape[3];

  int64_t output_height = y_shape[2];
  int64_t output_width = y_shape[3];

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (channel + num_threads - 1) / num_threads;
  int64_t need_num_threads = (channel + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int start_channel = i * block_size;
    int end_channel = std::min<int64_t>(start_channel + block_size, channel);

    eigen_device->enqueue_with_barrier(
      &barrier,
      &max_unpooling2d_block_impl<T>,
      x,
      indices,
      y,
      start_channel,
      end_channel,
      input_height,
      input_width,
      output_height,
      output_width);
  }

  barrier.Wait();
}

void max_unpooling2d(const Tensor &x, const Tensor &indices, Tensor &y) {
  if (x.element_type().is<float>()) {
    max_unpooling2d_impl<float>(
      x.eigen_device().get(),
      x.data<float>(),
      x.shape(),
      indices.data<int64_t>(),
      indices.shape(),
      y.data<float>(),
      y.shape());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}
