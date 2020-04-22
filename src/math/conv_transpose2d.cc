#include "math/conv_transpose2d.h"

namespace dlfm {
namespace math {

template <typename T>
void conv_transpose2d_block_impl(T *mat,
                                 T *bias,
                                 T *output,
                                 int64_t input_channel,
                                 int64_t input_height,
                                 int64_t input_width,
                                 int64_t output_channel,
                                 int64_t start_output_channel,
                                 int64_t end_output_channel,
                                 int64_t output_height,
                                 int64_t output_width,
                                 int64_t kernel_height,
                                 int64_t kernel_width,
                                 int64_t stride_height,
                                 int64_t stride_width,
                                 int64_t pad_top,
                                 int64_t pad_left) {
  for (int64_t oy = 0; oy < output_height; ++oy) {
    int64_t iy = oy - pad_top;

    for (int64_t ox = 0; ox < output_width; ++ox) {
      int64_t ix = ox - pad_left;

      T *out = output + start_output_channel * output_height * output_width + oy * output_width + ox;

      for (int64_t oc = start_output_channel; oc < end_output_channel; ++oc) {
        out[0] = bias[oc];

        for (int64_t wy = 0; wy < kernel_height; ++wy) {
          int64_t r_iy = iy + wy;

          if (0 != (r_iy % stride_height)) {
            continue;
          }

          r_iy /= stride_height;

          if (r_iy < 0 || r_iy >= input_height) {
            continue;
          }

          // conv transpose2d need reverse weight
          int64_t r_wy = kernel_height - 1 - wy;

          for (int64_t wx = 0; wx < kernel_width; ++wx) {
            int64_t r_ix = ix + wx;

            if (0 != (r_ix % stride_width)) {
              continue;
            }

            r_ix /= stride_width;

            // conv transpose2d need reverse weight
            int64_t r_wx = kernel_width - 1 - wx;

            if (0 <= r_ix && r_ix < input_width) {
              // mat shape [input_height, intput_width, output_channel, kernel_height, kernel_width]
              out[0] += mat[r_iy * input_width * output_channel * kernel_height * kernel_width + 
                            r_ix * output_channel * kernel_height * kernel_width + 
                              oc * kernel_height * kernel_width + 
                            r_wy * kernel_width +
                            r_wx];
            }
          }
        }

        out += output_height * output_width;
      }
    }
  }
}

template <typename T>
void conv_transpose2d_impl(Eigen::ThreadPoolDevice *eigen_device,
                           T *mat,
                           T *bias,
                           T *output,
                           int64_t input_channel,
                           int64_t input_height,
                           int64_t input_width,
                           int64_t output_channel,
                           int64_t output_height,
                           int64_t output_width,
                           int64_t kernel_height,
                           int64_t kernel_width,
                           int64_t stride_height,
                           int64_t stride_width,
                           int64_t pad_top,
                           int64_t pad_left) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();

  int64_t block_size = (output_channel + num_threads - 1) / num_threads;
  int64_t need_num_threads = (output_channel + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int start_channel = i * block_size;
    int end_channel = std::min<int64_t>(start_channel + block_size, output_channel);

    eigen_device->enqueue_with_barrier(
      &barrier,
      &conv_transpose2d_block_impl<T>,
      mat,
      bias,
      output,
      input_channel,
      input_height,
      input_width,
      output_channel,
      start_channel,
      end_channel,
      output_height,
      output_width,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_top,
      pad_left);
  }

  barrier.Wait();
}

void conv_transpose2d(const Tensor &input,
                      const Tensor &weight,
                      const Tensor &bias,
                      Tensor &output,
                      std::vector<size_t> kernel_size,
                      std::vector<size_t> stride,
                      std::vector<size_t> padding,
                      std::vector<size_t> out_padding) {
  // input [batch, input_channel, input_height, intput_width] (batch must be 1) -> [input_channel, input_height * intput_width ]
  // weight [input_channel, output_channel, kernel_height, kernel_width] -> [input_channel, output_channel * kernel_height * kernel_width]
  // input.t X weight -> [input_height * intput_width, output_channel * kernel_height * kernel_width]
  // extract output from input.t X weight
  ARGUMENT_CHECK(1 == input.shape()[0] && 1 == output.shape()[0], "conv_transpose2d need batch is 1");
  ARGUMENT_CHECK(input.element_type().is<float>(), "element type:" << input.element_type().name() << " nor support!");

  auto input_channel = input.shape()[1];
  auto input_height = input.shape()[2];
  auto input_width = input.shape()[3];

  auto output_channel = output.shape()[1];
  auto output_height = output.shape()[2];
  auto output_width = output.shape()[3];

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width  = kernel_size[1];
  int64_t stride_height = stride[0];
  int64_t stride_width  = stride[1];

  int64_t pad_top  = kernel_size[0] - padding[0] - 1;
  int64_t pad_left = kernel_size[1] - padding[1] - 1;

  auto mat = input.reshape({ input_channel, input_height * input_width })
    .matmul(weight.reshape({ input_channel, output_channel * kernel_height * kernel_width }), true, false);

  // mat [input_height, input_width, output_channel, kernel_height, kernel_width]
  conv_transpose2d_impl<float>(
    input.eigen_device().get(),
    mat.data<float>(),
    bias.data<float>(),
    output.data<float>(),
    input_channel,
    input_height,
    input_width,
    output_channel,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    pad_top,
    pad_left);
}

}
}