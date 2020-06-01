#ifdef HAS_NNPACK
#include "nnpack.h"
#endif

#include "math/conv2d.h"

namespace dlfm::math {

#ifdef HAS_NNPACK
// stride is 1x1 and group is 1
void conv2d_f32_stride1x1_group1_impl(
  const Tensor &input,
  const Tensor &weight,
  const Tensor &bias,
  Tensor &output,
  const std::vector<size_t> &padding) {
  float *input_ptr  = input.data<float>();
  float *weight_ptr = weight.data<float>();
  float *bias_ptr   = bias.data<float>();
  float *output_ptr = output.data<float>();

  int64_t batch = input.shape()[0];

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

  // for save memory, when the input image is too big use gemm
//  auto convolution_algorithm = nnp_convolution_algorithm_auto;
//  if (input_height * input_width * output_channel >= 512 * 512 * 64) {
//    convolution_algorithm = nnp_convolution_algorithm_implicit_gemm;
//  }
  auto convolution_algorithm = nnp_convolution_algorithm_auto;

  auto nnp_status = nnp_convolution_output(
    convolution_algorithm,
    batch,
    input_channel,
    output_channel,
    input_size,
    input_padding,
    kernel_size,
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    output.nnpack_threadpool(),
    nullptr);

  if (nnp_status != nnp_status_success) {
    std::cout << "nnp_convolution_output error:" << nnp_status << "\n";
  }
}

void conv2d_f32_batch1_group1_impl(
  const Tensor &input,
  const Tensor &weight,
  const Tensor &bias,
  Tensor &output,
  const std::vector<size_t> &stride,
  const std::vector<size_t> &padding) {
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

  // for save memory, when the input image is too big use gemm
  auto convolution_algorithm = nnp_convolution_algorithm_auto;
  if (input_height * input_width * output_channel >= 512 * 512 * 64) {
    convolution_algorithm = nnp_convolution_algorithm_implicit_gemm;
  }

  auto nnp_status = nnp_convolution_inference(
    convolution_algorithm,
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
    output.nnpack_threadpool(),
    nullptr);

  if (nnp_status != nnp_status_success) {
    std::cout << "nnp_convolution_inference error:" << nnp_status << "\n";
  }
}

void conv2d_f32_impl(
  const Tensor &input,
  const Tensor &weight,
  const Tensor &bias,
  Tensor &output,
  const std::vector<size_t> &stride,
  const std::vector<size_t> &padding,
  const size_t groups) {
  float *input_ptr  = input.data<float>();
  float *weight_ptr = weight.data<float>();
  float *bias_ptr   = bias.data<float>();
  float *output_ptr = output.data<float>();

  int64_t batch = input.shape()[0];

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

  int64_t input_group_channel = input_channel / groups;
  int64_t output_group_channel = output_channel / groups;

  // for save memory, when the input image is too big use gemm
  auto convolution_algorithm = nnp_convolution_algorithm_auto;
  if (input_height * input_width * input_group_channel >= 512 * 512 * 64) {
    convolution_algorithm = nnp_convolution_algorithm_implicit_gemm;
  }

  Eigen::ThreadPoolDevice *eigen_device = output.eigen_device().get();
  Eigen::Barrier barrier((unsigned int)(batch * groups));

  auto block = [] (
    enum nnp_convolution_algorithm algorithm,
    enum nnp_convolution_transform_strategy transform_strategy,
    size_t input_channels,
    size_t output_channels,
    struct nnp_size input_size,
    struct nnp_padding input_padding,
    struct nnp_size kernel_size,
    struct nnp_size output_subsampling,
    const float input[],
    const float kernel[],
    const float bias[],
    float output[],
    pthreadpool_t threadpool,
    struct nnp_profile* profile) {
      auto nnp_status = nnp_convolution_inference(
        algorithm,
        transform_strategy,
        input_channels,
        output_channels,
        input_size,
        input_padding,
        kernel_size,
        output_subsampling,
        input,
        kernel,
        bias,
        output,
        nullptr,
        nullptr);

    if (nnp_status != nnp_status_success) {
      std::cout << "nnp_convolution_inference error:" << nnp_status << "\n";
    }
  };

  for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
    for (int64_t g_idx = 0; g_idx < groups; ++g_idx) {
      auto idx = b_idx * groups + g_idx;

      eigen_device->enqueue_with_barrier(
        &barrier,
        block,
        convolution_algorithm,
        nnp_convolution_transform_strategy_block_based,
        input_group_channel,
        output_group_channel,
        input_size,
        input_padding,
        kernel_size,
        output_subsampling,
        input_ptr + idx * input_group_channel * input_height * input_width,
        weight_ptr + g_idx * output_group_channel * input_group_channel * kernel_height * kernel_width,
        bias_ptr + g_idx * output_group_channel,
        output_ptr + idx * output_group_channel * output_height * output_width,
        nullptr,
        nullptr);
    }
  }

  barrier.Wait();
}

#endif

void conv2d(
  const Tensor &input,
  const Tensor &weight,
  const Tensor &bias,
  Tensor &output,
  std::vector<size_t> stride,
  std::vector<size_t> padding,
  size_t groups) {
  ARGUMENT_CHECK(input.element_type().is<float>(), "conv2d only support float");

#ifdef HAS_NNPACK
  if (1 == input.shape()[0] && 1 == groups) {
    conv2d_f32_batch1_group1_impl(
            input,
            weight,
            bias,
            output,
            stride,
            padding);
  } else if (1 == stride[0] && 1 == stride[1] && 1 == groups) {
    conv2d_f32_stride1x1_group1_impl(
      input,
      weight,
      bias,
      output,
      padding);
  } else {
    conv2d_f32_impl(
      input,
      weight,
      bias,
      output,
      stride,
      padding,
      groups);
  }
#else
  RUNTIME_ERROR("please build with nnpack");
#endif
}

}
