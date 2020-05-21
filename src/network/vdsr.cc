#include "common/tensor.h"
#include "module/conv2d.h"
#include "module/relu.h"
#include "module/sequential.h"
#include "network/vdsr.h"

namespace dlfm::nn::vdsr {

ConvReluBlock::ConvReluBlock() {
  ADD_SUB_MODULE(conv, conv2d, 64, 64, 3, 1, 1, 1, false);
  ADD_SUB_MODULE(relu, std::make_shared<ReluImpl>, true);
}

Tensor ConvReluBlock::forward(Tensor x) {
  return (*relu)((*conv)(x));
}

VDSR::VDSR() {
  std::vector<Module> residual_layer_nodes;

  for (int64_t i = 0; i < 18; ++i) {
    residual_layer_nodes.emplace_back(std::make_shared<ConvReluBlock>());
  }

  ADD_SUB_MODULE(residual_layer, sequential, residual_layer_nodes);

  ADD_SUB_MODULE(input, conv2d, 1, 64, 3, 1, 1, 1, false);
  ADD_SUB_MODULE(output, conv2d, 64, 1, 3, 1, 1, 1, false);
}

/*  JPEG/JFIF YCbCr conversions

    Y  = R *  0.29900 + G *  0.58700 + B *  0.11400
    Cb = R * -0.16874 + G * -0.33126 + B *  0.50000 + 128
    Cr = R *  0.50000 + G * -0.41869 + B * -0.08131 + 128

    R  = Y + (Cb - 128) * 0        + (Cr - 128) *  1.40200
    G  = Y + (Cb - 128) * -0.34414 + (Cr - 128) * -0.71414
    B  = Y + (Cb - 128) *  1.77200 + (Cr - 128) * 0

*/
// convert rgb[float] to ycbcr[float] [0-255]
Tensor VDSR::rgb_2_ycbcr(Tensor &rgb) {
  // rgb float [0-255] [h, w, 3]
  auto h = rgb.shape()[0];
  auto w = rgb.shape()[1];

  float weight_v[] = { 0.29900, -0.16874 , 0.50000, 0.58700 , -0.33126 , -0.41869 , 0.11400 , 0.50000 , -0.08131 };
  float bias_v[] = {0, 128, 128};

  auto weight = Tensor::create_from(weight_v, {3, 3,}, ElementType::from<float>());
  auto bias = Tensor::create_from(bias_v, { 3, }, ElementType::from<float>());

  auto ycbcr = rgb.reshape({h * w, 3}).matmul(weight) + bias;

  return ycbcr.reshape({h, w, 3});
}

Tensor VDSR::ycbcr_2_rgb(Tensor &ycbcr) {
  auto h = ycbcr.shape()[0];
  auto w = ycbcr.shape()[1];

  float bias_v[] = { 0, -128, -128 };
  float weight_v[] = { 1, 1 , 1 , 0 , -0.34414 , 1.77200 , 1.40200 , -0.71414 , 0 };

  auto bias = Tensor::create_from(bias_v, { 3, }, ElementType::from<float>());
  auto weight = Tensor::create_from(weight_v, { 3, 3, }, ElementType::from<float>());

  auto rgb = (ycbcr.reshape({h * w, 3}) + bias).matmul(weight);

  return rgb.reshape({ h, w, 3 });
}

Tensor VDSR::forward(Tensor x) {
  auto residual = x;

  auto out = (*input)(x).relu(true);
  out = (*residual_layer)(out);
  out = (*output)(out);

  return out + residual;
}

}