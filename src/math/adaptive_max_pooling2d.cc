#include "common/basic.h"
#include "math/adaptive_max_pooling2d.h"

namespace dlfm::math {

template <typename T>
void adaptive_max_pooling2d_block_impl(
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
  float stride_y,
  float stride_x) {
  for (int64_t c = start_channel; c < end_channel; ++c) {
    for (int64_t oy = 0; oy < output_height; ++oy) {
      for (int64_t ox = 0; ox < output_width; ++ox) {
        T max_val = T(0);

        int64_t s_iy = (int64_t)std::round(float(oy) * stride_y);
        int64_t s_ix = (int64_t)std::round(float(ox) * stride_x);

        for (int64_t iy = s_iy; iy < s_iy + kernel_height; ++iy) {
          for (int64_t ix = s_ix; ix < s_ix + kernel_width; ++ix) {
            max_val = std::max<T>(max_val, input[c * input_height * input_width + iy * input_width + ix]);
          }
        }

        output[c * output_height * output_width + oy * output_width + ox] = max_val;
      }
    }
  }
}

template <typename T>
void adaptive_max_pooling2d_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  T *x,
  const Shape &xshape,
  T *y,
  const Shape &yshape) {
  auto channel = xshape[0] * xshape[1];
  int64_t input_height = xshape[2];
  int64_t input_width  = xshape[3];

  int64_t output_height = yshape[2];
  int64_t output_width  = yshape[3];

  auto kernel_height = (input_height + output_height - 1) / output_height;
  auto kernel_width = (input_width + output_width - 1) / output_width;

  float stride_y = 0;
  float stride_x = 0;

  if (output_height > 1) {
    stride_y = float(input_height - kernel_height) / float(output_height - 1);
  }

  if (output_width > 1) {
    stride_x = float(input_width - kernel_width) / float(output_width - 1);
  }

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size  = (channel + num_threads - 1) / num_threads;

  num_threads = (channel + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int start_channel = i * block_size;
    int end_channel = std::min<int64_t>(start_channel + block_size, channel);

    eigen_device->enqueue_with_barrier(
        &barrier,
        &adaptive_max_pooling2d_block_impl<T>,
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
        stride_x);
  }

  barrier.Wait();
}

void adaptive_max_pooling2d(const Tensor &x, Tensor &y) {
  if (x.element_type().is<float>()) {
    adaptive_max_pooling2d_impl<float>(
      x.eigen_device().get(),
      x.data<float>(),
      x.shape(),
      y.data<float>(),
      y.shape());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}


}