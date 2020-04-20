#ifdef HAS_NNPACK
#include "nnpack.h"
#endif

#include "math/conv2d.h"

namespace dlfm::math {

#ifdef HAS_NNPACK
// batch must be 1
void conv2d_float_impl(
  const Tensor &input,
  const Tensor &weight,
  const Tensor &bias,
  Tensor &output,
  const std::vector<int64_t> &stride,
  const std::vector<int64_t> &padding,
  const int64_t groups) {
  auto eigen_device = output.eigen_device().get(),

  float *input_ptr  = input.data<float>();
  float *weight_ptr = weight.data<float>();
  float *bias_ptr   = bias.data<float>();
  float *output_ptr = output.data<float>();

  int64_t input_channel = input.shape()[-3];
  int64_t input_height  = input.shape()[-2];
  int64_t input_width   = input.shape()[-1];

  int64_t output_channel = output.shape()[-3];
  int64_t output_height  = output.shape()[-2];
  int64_t output_width   = output.shape()[-1];

  int64_t kernel_height = weight.shape()[2];
  int64_t kernel_width = weight.shape()[3];

  struct nnp_size input_size;
  input_size.width  = (size_t)input_width;
  input_size.height = (size_t)input_height;

  struct nnp_padding input_padding;
  input_padding.top = padding[0];
  input_padding.right = padding[1];
  input_padding.bottom = padding[0];
  input_padding.left = padding[1];

  struct nnp_size kernel_size;
  kernel_size.width = (size_t)kernel_width;
  kernel_size.height = (size_t)kernel_height;

  struct nnp_size output_subsampling;
  output_subsampling.width = stride[1];
  output_subsampling.height = stride[0];

  if (1 == groups) {
    // use nnpack directly
    auto nnp_status = nnp_convolution_inference(
      nnp_convolution_algorithm_auto,
      nnp_convolution_transform_strategy_block_based,
      input_channel,
      output_channel,
      input_size,
      input_padding,
      kernel_size,
      output_subsampling,
      input_ptr,
      weight_ptr,
      bias_ptr,
      output_ptr,
      // set threadpool
      output.nnpack_threadpool(),
      nullptr);
  } else {
    // multi-thread calculate result
    Eigen::Barrier barrier((unsigned int)(groups));

    auto block = [&barrier, &input_size, &input_padding, &kernel_size, &output_subsampling] (
      int64_t input_channel,
      int64_t output_channel,
      float *input_ptr,
      float *weight_ptr,
      float *bias_ptr,
      float *output_ptr) {
      auto nnp_status = nnp_convolution_inference(
        nnp_convolution_algorithm_auto,
        nnp_convolution_transform_strategy_block_based,
        input_channel,
        output_channel,
        input_size,
        input_padding,
        kernel_size,
        output_subsampling,
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        // threadpool set nullptr
        nullptr,
        nullptr);

      barrier.Notify();
    };

    int64_t input_channel_step = input_channel / groups;
    int64_t output_channel_step = output_channel / groups;

    for (int64_t g = 0; g < groups; ++g) {
      eigen_device->enqueueNoNotification(
        block,
        input_channel_step,
        output_channel_step,
        input_ptr  + g * input_channel_step * input_height * input_width,
        weight_ptr + g * output_channel_step * input_channel_step * kernel_height * kernel_width,
        bias_ptr   + g * output_channel_step,
        output_ptr + g * output_channel_step * output_height * output_width);
    }

    barrier.Wait();
  }
}
#endif

void conv2d(const Tensor &input, const Tensor &weight, const Tensor &bias, Tenor &output, std::vector<size_t> stride, std::vector<size_t> padding, int64_t groups) {
  ARGUMENT_CHECK(1 == input.shape()[0] && 1 == output.shape()[0], "conv2d need batch is 1");

  if (input.element_type().is<float>()) {
#ifdef HAS_NNPACK
    conv2d_float_impl(input, weight, bias, output, stride, padding, groups);
#else
    RUNTIME_ERROR("please build with nnpack");
#endif
  } else {
    RUNTIME_ERROR("element type:" << input.element_type().name() << " nor support!");
  }
}

}
