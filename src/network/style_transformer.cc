#include "common/cost_helper.h"
#include "module/reflection_pad2d.h"
#include "network/style_transformer.h"

namespace dlfm::nn::style_transformer {

ConvBlock::ConvBlock(
  int64_t in_channels,
  int64_t out_channels,
  int64_t kernel_size,
  int64_t stride,
  bool will_upsample,
  bool will_normalize,
  bool will_relu) {
  upsample = will_upsample;
  relu = will_relu;

  ADD_SUB_MODULE(conv, conv2d, in_channels, out_channels, kernel_size, stride, kernel_size/2);

  if (will_normalize) {
    ADD_SUB_MODULE(norm, instance_norm2d, out_channels, 1e-05, true);
  }
}

Tensor ConvBlock::forward(Tensor x) {
  if (upsample) {
    CostHelper::start("upsample2d");
    x = x.upsample2d(2, "bilinear");
    CostHelper::end();
  }

  CostHelper::start("conv");
  x = (*conv)(x);
  CostHelper::end();

  if (norm) {
    CostHelper::start("norm");
    x = (*norm)(x);
    CostHelper::end();
  }

  if (relu) {
    CostHelper::start("relu");
    x.relu(true);
    CostHelper::end();
  }

  return x;
}

ResidualBlock::ResidualBlock(int64_t channels) {
  ADD_SUB_MODULE(block, sequential, {
    std::make_shared<ConvBlock>(channels, channels, 3, 1, false, true, true),
    std::make_shared<ConvBlock>(channels, channels, 3, 1, false, true, false)
  });
}

Tensor ResidualBlock::forward(Tensor x) {
  return (*block)(x) + x;
}

Transformer::Transformer() {
  ADD_SUB_MODULE(model, sequential, {
    std::make_shared<ConvBlock>(3, 32, 5, 1),
    std::make_shared<ConvBlock>(32, 64, 3, 2),
    std::make_shared<ConvBlock>(64, 128, 3, 2),
    std::make_shared<ResidualBlock>(128),
    std::make_shared<ResidualBlock>(128),
    std::make_shared<ResidualBlock>(128),
    std::make_shared<ResidualBlock>(128),
    std::make_shared<ResidualBlock>(128),
    std::make_shared<ConvBlock>(128, 64, 3, 1, true),
    std::make_shared<ConvBlock>(64, 32, 3, 1, true),
    std::make_shared<ConvBlock>(32, 3, 5, 1, false, false, false),
  });

  model->print_log_ = true;

  mean = Tensor::create({3});
  std = Tensor::create({3});

  float *mean_ptr = mean.data<float>();
  float *std_ptr = std.data<float>();

  // mean = np.array([0.485, 0.456, 0.406])
  // std = np.array([0.229, 0.224, 0.225])
  mean_ptr[0] = 0.485;
  mean_ptr[1] = 0.456;
  mean_ptr[2] = 0.406;

  std_ptr[0] = 0.229;
  std_ptr[1] = 0.224;
  std_ptr[2] = 0.225;
}

Tensor Transformer::forward(Tensor x) {
  // x is [h, w, 3] uint8 image
  ARGUMENT_CHECK(x.element_type().is<uint8_t>(), "style Transformer need uint8 input");
  ARGUMENT_CHECK(3 == x.shape().ndims() && 3 == x.shape()[2], "style Transformer input shape error");
  ARGUMENT_CHECK(x.shape()[0] >= 32 && 0 == x.shape()[0] % 4 && x.shape()[1] >= 32 && 0 == x.shape()[1] % 4,  "style Transformer input shape error");
  
  // -> [3, h, w]
  x = x.transpose({2, 0, 1});
  
  // float [3, h, w]
  x = x.cast(ElementType::from<float>());
  x *= (1 / 255.0);

  // normalize
  x = x.normalize(mean, std, true);

  // transform
  // [1, 3, h, w]
  x = (*model)(x.unsqueeze(0));

  // [3, h, w]
  x = x.squeeze(0);

  // [3, h, w]
  x = x.denormalize(mean, std, true);

  // cast to uint8
  x *= 255.0;
  x = x.clamp(0, 255, true).cast(ElementType::from<uint8_t>());

  return x.transpose({1, 2, 0});
}

}