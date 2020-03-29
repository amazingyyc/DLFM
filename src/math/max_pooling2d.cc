#include "common/basic.h"
#include "math/max_pooling2d.h"

namespace dlfm::math {

//template <typename T>
//void max_pooling2d_impl(Eigen::ThreadPoolDevice *eigen_device,
//                        T *x,
//                        const Shape &xshape,
//                        T *y,
//                        const Shape &yshape,
//                        std::vector<size_t> kernel_size,
//                        std::vector<size_t> stride,
//                        std::vector<size_t> padding) {
//  int64_t batch = xshape[0];
//  int64_t channel = xshape[1];
//
//  int64_t input_height = xshape[2];
//  int64_t input_width  = xshape[3];
//
//  int64_t output_height = yshape[2];
//  int64_t output_width  = yshape[3];
//
//  // xshape [batch_size, channel, input_height, input_width]
//  // yshape [batch_size, channel, output_height, output_width]
//  Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> xvec(x, batch, channel, input_height, input_width);
//  Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> yvec(y, batch, channel, output_height, output_width);
//
//  // x -> [batch_size, input_height, input_width, channel]
//  Eigen::array<Eigen::Index, 4> x_shuffle = { 0, 2, 3, 1 };
//  Eigen::array<Eigen::Index, 4> extract_reshape = {batch * output_height * output_width, (int64_t)kernel_size[0], (int64_t)kernel_size[1], channel };
//  Eigen::array<Eigen::Index, 2> maximum_axis = {1, 2};
//  Eigen::array<Eigen::Index, 4> maximum_reshape = { batch, output_height, output_width, channel };
//  Eigen::array<Eigen::Index, 4> y_shuffle = { 0, 3, 1, 2 };
//
//  Eigen::Index  padding_top    = padding[1];
//  Eigen::Index  padding_bottom = (output_width  - 1) * stride[1] + kernel_size[1] - input_width  - padding_top;
//  Eigen::Index  padding_left   = padding[0];
//  Eigen::Index  padding_right  = (output_height - 1) * stride[0] + kernel_size[0] - input_height - padding_left;
//
//  yvec.device(*eigen_device) = xvec.shuffle(x_shuffle)
//      .extract_image_patches(kernel_size[1], kernel_size[0],
//                             stride[1], stride[0],
//                             1, 1,
//                             1, 1,
//                             padding_top, padding_bottom,
//                             padding_left, padding_right,
//                             0)
//      .reshape(extract_reshape)
//      .maximum(maximum_axis)
//      .reshape(maximum_reshape)
//      .shuffle(y_shuffle);
//}

template <typename T>
void max_pooling2d_block_impl(T *input,
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
        T max_val = T(0);

        for (int64_t i_y = o_y * stride_y - pad_top; i_y < o_y * stride_y - pad_top + kernel_height; ++i_y) {
          for (int64_t i_x = o_x * stride_x - pad_left; i_x < o_x * stride_x - pad_left + kernel_width; ++i_x) {
            if (i_y >= 0 && i_y < input_height && i_x >= 0 && i_x < input_width) {
              max_val = std::max<T>(max_val, input[c * input_height * input_width + i_y * input_width + i_x]);
            }
          }
        }

        output[c * output_height * output_width + o_y * output_width + o_x] = max_val;
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

  auto block = [&barrier](T *x,
                          T *y,
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
    max_pooling2d_block_impl(
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

    barrier.Notify();
  };

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int start_channel = i * block_size;
    int end_channel = std::min<int64_t>(start_channel + block_size, channel);

    eigen_device->enqueueNoNotification(
        block,
        x, y,
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
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}


}