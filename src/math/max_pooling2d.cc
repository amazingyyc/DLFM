#include "common/basic.h"
#include "math/max_pooling2d.h"

namespace dlfm::math {

template <typename T>
void max_pooling2d_block_impl(
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
    for (int64_t oy = 0; oy < output_height; ++oy) {
      for (int64_t ox = 0; ox < output_width; ++ox) {
        T max_val = std::numeric_limits<T>::lowest();

        int64_t iy_start = oy * stride_y - pad_top;
        int64_t ix_start = ox * stride_x - pad_left;
        int64_t iy_end = oy * stride_y - pad_top + kernel_height;
        int64_t ix_end = ox * stride_x - pad_left + kernel_width;

        if (iy_start < 0 || ix_start < 0 || iy_end > input_height || ix_end > input_width) {
          max_val = 0;
        }

        for (int64_t iy = std::max<int64_t>(0, iy_start); iy < std::min(iy_end, input_height); ++iy) {
          for (int64_t ix = std::max<int64_t>(0, ix_start); ix < std::min(ix_end, input_width); ++ix) {
            max_val = std::max<T>(max_val, input[c * input_height * input_width + iy * input_width + ix]);
          }
        }

        output[c * output_height * output_width + oy * output_width + ox] = max_val;
      }
    }
  }
}

template <typename T>
void max_pooling2d_impl(Eigen::ThreadPoolDevice *eigen_device,
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
  int64_t need_num_threads = (channel + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int start_channel = i * block_size;
    int end_channel = std::min<int64_t>(start_channel + block_size, channel);

    eigen_device->enqueue_with_barrier(
        &barrier,
        &max_pooling2d_block_impl<T>,
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

void max_pooling2d(const Tensor &x, Tensor &y, std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding) {
  if (x.element_type().is<float>()) {
    max_pooling2d_impl<float>(
        x.eigen_device().get(),
        x.data<float>(),
        x.shape(),
        y.data<float>(),
        y.shape(),
        kernel_size,
        stride,
        padding);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

template <typename T>
void max_pooling2d_with_indices_block_impl(
  T *input,
  T *output,
  int64_t *indices,
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
    for (int64_t oy = 0; oy < output_height; ++oy) {
      for (int64_t ox = 0; ox < output_width; ++ox) {
        T max_val = std::numeric_limits<T>::lowest();
        int64_t max_idx = -1;

        int64_t iy_start = oy * stride_y - pad_top;
        int64_t ix_start = ox * stride_x - pad_left;
        int64_t iy_end = oy * stride_y - pad_top + kernel_height;
        int64_t ix_end = ox * stride_x - pad_left + kernel_width;

        if (iy_start < 0 || ix_start < 0 || iy_end > input_height || ix_end > input_width) {
          max_val = 0;
        }

        for (int64_t iy = std::max<int64_t>(0, iy_start); iy < std::min(iy_end, input_height); ++iy) {
          for (int64_t ix = std::max<int64_t>(0, ix_start); ix < std::min(ix_end, input_width); ++ix) {
            auto i_val = input[c * input_height * input_width + iy * input_width + ix];

            if (i_val > max_val) {
              max_val = i_val;
              max_idx = iy * input_width + ix;
            }
          }
        }

        output[c * output_height * output_width + oy * output_width + ox] = max_val;
        indices[c * output_height * output_width + oy * output_width + ox] = max_idx;
      }
    }
  }
}

template <typename T>
void max_pooling2d_with_indices_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  T *x,
  const Shape &x_shape,
  T *y,
  const Shape &y_shape,
  int64_t *indices,
  const Shape &indices_shape,
  const std::vector<size_t> &kernel_size,
  const std::vector<size_t> &stride,
  const std::vector<size_t> &padding) {
  int64_t channel = x_shape[0] * x_shape[1];
  int64_t input_height = x_shape[2];
  int64_t input_width = x_shape[3];

  int64_t output_height = y_shape[2];
  int64_t output_width = y_shape[3];

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t stride_y = stride[0];
  int64_t stride_x = stride[1];
  int64_t pad_top = padding[0];
  int64_t pad_left = padding[1];

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (channel + num_threads - 1) / num_threads;
  int64_t need_num_threads = (channel + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int start_channel = i * block_size;
    int end_channel = std::min<int64_t>(start_channel + block_size, channel);

    eigen_device->enqueue_with_barrier(
      &barrier,
      &max_pooling2d_with_indices_block_impl<T>,
      x,
      y,
      indices,
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

void max_pooling2d_with_indices(const Tensor &x, Tensor &y, Tensor &indices, std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding) {
  ARGUMENT_CHECK(indices.element_type().is<int64_t>(), "indices element_type must be int64");

  if (x.element_type().is<float>()) {
    max_pooling2d_with_indices_impl<float>(
      x.eigen_device().get(),
      x.data<float>(),
      x.shape(),
      y.data<float>(),
      y.shape(),
      indices.data<int64_t>(),
      indices.shape(),
      kernel_size,
      stride,
      padding);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}