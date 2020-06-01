#include "common/tensor.h"
#include "module/conv2d.h"
#include "module/sequential.h"
#include "module/batch_norm2d.h"
#include "module/relu6.h"
#include "module/dropout.h"
#include "network/human_seg.h"

#include <utility>

namespace dlfm::nn::human_seg {

InvertedResidual::InvertedResidual(int64_t inp, int64_t oup, int64_t s, int64_t expansion, int64_t dilation) {
  ARGUMENT_CHECK(1 == s || 2 == s, "stride must be 1 or 2");
  ARGUMENT_CHECK(1 == dilation, "InvertedResidual need dilation is 1");

  stride = s;
  use_res_connect = ((stride == 1) && (inp == oup));

  auto hidden_dim = inp * expansion;

  if (1 == expansion) {
    ADD_SUB_MODULE(conv, sequential, {
      conv2d(hidden_dim, hidden_dim, 3, stride, 1, hidden_dim, false),
      batch_norm2d(hidden_dim, 1e-5),
      relu6(true),
      conv2d(hidden_dim, oup, 1, 1, 0, 1, false),
      batch_norm2d(oup, 1e-5)
    });
  } else {
    ADD_SUB_MODULE(conv, sequential, {
      conv2d(inp, hidden_dim, 1, 1, 0, 1, false),
      batch_norm2d(hidden_dim, 1e-5),
      relu6(true),
      conv2d(hidden_dim, hidden_dim, 3, stride, 1, hidden_dim, false),
      batch_norm2d(hidden_dim, 1e-5),
      relu6(true),
      conv2d(hidden_dim, oup, 1, 1, 0, 1, false),
      batch_norm2d(oup, 1e-5)
    });
  }
}

Tensor InvertedResidual::forward(Tensor x) {
  if (use_res_connect) {
    return x + (*conv)(x);
  } else {
    return (*conv)(x);
  }
}

MobileNetV2::MobileNetV2(float alpha, int64_t expansion, int64_t classes) {
  num_classes = classes;

  int64_t input_channel = 32;
  int64_t l_channel = 1280;

  std::vector<std::vector<int64_t>> interverted_residual_setting = {
          {1        , 16, 1, 1},
          {expansion, 24, 2, 2},
          {expansion, 32, 3, 2},
          {expansion, 64, 4, 2},
          {expansion, 96, 3, 1},
          {expansion, 160, 3, 2},
          {expansion, 320, 1, 1},
  };

  input_channel = make_divisible(input_channel * alpha, 8);
  last_channel = alpha > 1.0 ? make_divisible(l_channel * alpha, 8) : l_channel;

  std::vector<Module> features_ndoes;

  features_ndoes.emplace_back(conv_bn(3, input_channel, 2));

  for (auto setting : interverted_residual_setting) {
    auto t = setting[0];
    auto c = setting[1];
    auto n = setting[2];
    auto s = setting[3];

    int64_t output_channel = make_divisible(c * alpha, 8);

    for (int64_t i = 0; i < n; ++i) {
      if (0 == i) {
        features_ndoes.emplace_back(std::make_shared<InvertedResidual>(input_channel, output_channel, s, t));
      } else {
        features_ndoes.emplace_back(std::make_shared<InvertedResidual>(input_channel, output_channel, 1, t));
      }

      input_channel = output_channel;
    }
  }

  features_ndoes.emplace_back(conv_1x1_bn(input_channel, last_channel));

  ADD_SUB_MODULE(features, sequential, features_ndoes);

  if (num_classes > 0) {
    ADD_SUB_MODULE(classifier, sequential, {
      dropout(),
      linear(last_channel, num_classes)
    });
  }
}

int64_t MobileNetV2::make_divisible(float v, int64_t divisor, int64_t min_value) {
  if (min_value < 0) {
    min_value = divisor;
  }

  int64_t new_v = std::max(min_value, int64_t(v + divisor / 2.f) / divisor * divisor);

  if (float(new_v) < 0.9f * v) {
    new_v += divisor;
  }

  return new_v;
}

Sequential MobileNetV2::conv_bn(int64_t inp, int64_t oup, int64_t stride) {
  return sequential({
    conv2d(inp, oup, 3, stride, 1, 1, false),
    batch_norm2d(oup, 1e-5),
    relu6(true)
  });
}

Sequential MobileNetV2::conv_1x1_bn(int64_t inp, int64_t oup) {
  return sequential({
    conv2d(inp, oup, 1, 1, 0, 1, false),
    batch_norm2d(oup, 1e-5),
    relu6(true)
  });
}

Tensor MobileNetV2::forward(Tensor x) {
  for (size_t i = 0; i < 19; ++i) {
    x = (*(features->sub_modules_[i]))(x);
  }

  if (num_classes > 0) {
    x = x.mean({2, 3});
    x = (*classifier)(x);
  }

  return x;
}

DecoderBlock::DecoderBlock(int64_t in_channels, int64_t out_channels, std::shared_ptr<InvertedResidual> block) {
  ADD_SUB_MODULE(deconv, conv_tranpose2d, in_channels, out_channels, 4, 2, 1);
  ADD_SUB_MODULE(block_unit, std::move, block);
}

Tensor DecoderBlock::forward(std::vector<Tensor> inputs) {
  auto input = inputs[0];
  auto shortcut = inputs[1];

  auto x = (*deconv)(input);
  x = x.cat(shortcut, 1);
  x = (*block_unit)(x);

  return x;
}

HumanSeg::HumanSeg(int64_t num_classes) {
  float alpha = 1.0;
  int64_t expansion = 6;

  ADD_SUB_MODULE(backbone, std::make_shared<MobileNetV2>, alpha, expansion, -1);

  // Stage 1
  int64_t channel1 = backbone->make_divisible(96 * alpha, 8);
  auto block_unit1 = std::make_shared<InvertedResidual>(2 * channel1, channel1, 1, expansion);
  ADD_SUB_MODULE(decoder1, std::make_shared<DecoderBlock>, backbone->last_channel, channel1, block_unit1);

  // Stage 2
  auto channel2 = backbone->make_divisible(32 * alpha, 8);
  auto block_unit2 = std::make_shared<InvertedResidual>(2 * channel2, channel2, 1, expansion);
  ADD_SUB_MODULE(decoder2, std::make_shared<DecoderBlock>, channel1, channel2, block_unit2);

  // stage 3
  auto channel3 = backbone->make_divisible(24 * alpha, 8);
  auto block_unit3 = std::make_shared<InvertedResidual>(2 * channel3, channel3, 1, expansion);
  ADD_SUB_MODULE(decoder3, std::make_shared<DecoderBlock>, channel2, channel3, block_unit3);

  // stage 4
  auto channel4 = backbone->make_divisible(16 * alpha, 8);
  auto block_unit4 = std::make_shared<InvertedResidual>(2 * channel4, channel4, 1, expansion);
  ADD_SUB_MODULE(decoder4, std::make_shared<DecoderBlock>, channel3, channel4, block_unit4);

  ADD_SUB_MODULE(conv_last, sequential, {
    conv2d(channel4, 3, 3, 1, 1),
    conv2d(3, num_classes, 3, 1, 1),
  });

  mean = Tensor::create({ 3 });
  std = Tensor::create({ 3 });

  float *mean_ptr = mean.data<float>();
  float *std_ptr = std.data<float>();

  mean_ptr[0] = 0.485;
  mean_ptr[1] = 0.456;
  mean_ptr[2] = 0.406;

  std_ptr[0] = 0.229;
  std_ptr[1] = 0.224;
  std_ptr[2] = 0.225;
}

Tensor HumanSeg::forward(Tensor input) {
  // input must [3, h, w] and float [0-255]
  ARGUMENT_CHECK(3 == input.ndims() && input.element_type().is<float>(), "HumanSeg input error");
  ARGUMENT_CHECK(0 == input.shape()[1] % 2 && 0 == input.shape()[2] % 2 && 3 == input.shape()[0], "HumanSeg shape error");

  int64_t height = input.shape()[1];
  int64_t width = input.shape()[2];

  input = input * (1 / 255.0);

  // normalize
  input = input.normalize(mean, std, true);

  auto mobilenetv2 = backbone->features->sub_modules();

  // [1, 3, height, width]
  auto x = input.unsqueeze(0);

  for (int i = 0; i < 2; ++i) {
    x = (*(mobilenetv2[i]))(x);
  }

  auto x1 = x;

  for (int i = 2; i < 4; ++i) {
    x = (*(mobilenetv2[i]))(x);
  }

  auto x2 = x;

  for (int i = 4; i < 7; ++i) {
    x = (*(mobilenetv2[i]))(x);
  }

  auto x3 = x;

  for (int i = 7; i < 14; ++i) {
    x = (*(mobilenetv2[i]))(x);
  }

  auto x4 = x;

  for (int i = 14; i < 19; ++i) {
    x = (*(mobilenetv2[i]))(x);
  }

  auto x5 = x;

  x = (*decoder1)({x5, x4});
  x = (*decoder2)({x, x3});
  x = (*decoder3)({x, x2});
  x = (*decoder4)({x, x1});
  x = (*conv_last)(x);

  x = x.interpolate2d({ height, width }, "bilinear", true);
  x = x.reshape({2, height, width});

  // [2, height, width]
  x = x.softmax(0);

  // fetch [1, ... ]
  return x[1];
}

}






























