#include "common/basic.h"
#include "math/avg_pooling2d.h"

namespace dlfm::math {

template <typename T>
void avg_pooling2d_block_impl(
  T *input,
  T *output,
  int64_t start_channel,
  int64_t end_channel,
  int64_t input_height,
  int64_t input_width,
  int64_t output_height,
  int64_t output_width,
  int64_t kernel_height,
  int64_t kernel_width,
  int64_t stride_y,
  int64_t stride_x,
  int64_t pad_top,
  int64_t pad_left) {
  for (int64_t c = start_channel; c < end_channel; ++c) {
    for (int64_t o_y = 0; o_y < output_height; ++o_y) {
      for (int64_t o_x = 0; o_x < output_width; ++o_x) {
        T sum_val = T(0);

        for (int64_t i_y = o_y * stride_y - pad_top; i_y < o_y * stride_y - pad_top + kernel_height; ++i_y) {
          for (int64_t i_x = o_x * stride_x - pad_left; i_x < o_x * stride_x - pad_left + kernel_width; ++i_x) {
            if (i_y >= 0 && i_y < input_height && i_x >= 0 && i_x < input_width) {
              sum_val += input[c * input_height * input_width + i_y * input_width + i_x];
            }
          }
        }

        output[c * output_height * output_width + o_y * output_width + o_x] = sum_val / T(kernel_height * kernel_width);
      }
    }
  }
}

template <typename T>
void avg_pooling2d_impl(Eigen::ThreadPoolDevice *eigen_device,
                        T *x,
                        const Shape &x_shape,
                        T *y,
                        const Shape &y_shape,
                        const std::vector<size_t> &kernel_size,
                        const std::vector<size_t> &stride,
                        const std::vector<size_t> &padding) {
  int64_t channel      = x_shape[0] * x_shape[1];
  int64_t input_height = x_shape[2];
  int64_t input_width  = x_shape[3];

  int64_t output_height = y_shape[2];
  int64_t output_width  = y_shape[3];

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t stride_y = stride[0];
  int64_t stride_x = stride[1];
  int64_t pad_top  = padding[0];
  int64_t pad_left = padding[1];

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size  = (channel + num_threads - 1) / num_threads;

  num_threads = (channel + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int start_channel = i * block_size;
    int end_channel = std::min<int64_t>(start_channel + block_size, channel);

    eigen_device->enqueue_with_barrier(
        &barrier,
        &avg_pooling2d_block_impl<T>,
        x,
        y,
        start_channel,
        end_channel,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_y,
        stride_x,
        pad_top,
        pad_left);
  }

  barrier.Wait();
}

void avg_pooling2d(const Tensor &x, Tensor &y, std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding) {
  if (x.element_type().is<float>()) {
    avg_pooling2d_impl<float>(
        x.eigen_device().get(),
        x.data<float>(),
        x.shape(),
        y.data<float>(),
        y.shape(),
        kernel_size,
        stride,
        padding);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}


}