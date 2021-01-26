#include "common/tensor.h"
#include "module/conv2d.h"
#include "module/relu.h"
#include "module/tanh.h"
#include "module/sequential.h"
#include "module/upsample2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/conv_transpose2d.h"
#include "module/batch_norm2d.h"
#include "module/relu6.h"
#include "module/dropout.h"
#include "network/portrait_net.h"

namespace dlfm::nn::portrait_net {

InvertedResidual::InvertedResidual(int64_t inp, int64_t oup, int64_t s, int64_t expand_ratio, int64_t dilation) {
  ARGUMENT_CHECK(1 == s || 2 == s, "InvertedResidual stride error");
  ARGUMENT_CHECK(1 == dilation, "dilation must be 1");

  stride = s;
  use_res_connect = ((1 == stride) && (inp == oup));

  ADD_SUB_MODULE(conv, sequential, {
    conv2d(inp, inp * expand_ratio, 1, 1, 0, 1, 1, false),
    batch_norm2d(inp * expand_ratio, 1e-05, true),
    relu(true),
    conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, 1, inp * expand_ratio, false),
    batch_norm2d(inp * expand_ratio, 1e-05, true),
    relu(true),
    conv2d(inp * expand_ratio, oup, 1, 1, 0, 1, 1, false),
    batch_norm2d(inp * expand_ratio, 1e-05, true),
  });
}

Tensor InvertedResidual::forward(Tensor x) {
  if (use_res_connect) {
    return x + (*conv)(x);
  } else {
    return (*conv)(x);
  }
}

ResidualBlock::ResidualBlock(int64_t inp, int64_t oup, int64_t stride) {
  ADD_SUB_MODULE(block, sequential, {
    conv_dw(inp, oup, 3, stride),
    conv2d(oup, oup, 3, 1, 1, 1, oup, false),
    batch_norm2d(oup, 1e-05, true),
    relu(true),
    conv2d(oup, oup, 1, 1, 0, 1, 1, false),
    batch_norm2d(oup, 1e-05, true),
  });

  if (inp != oup) {
    ADD_SUB_MODULE(residual, sequential, {
      conv2d(inp, oup, 1, 1, 0, 1, 1, false),
      batch_norm2d(oup, 1e-05, true),
    });
  }
}

Tensor ResidualBlock::forward(Tensor x) {
  auto res = x;

  auto out = (*block)(x);

  if (residual) {
    res = (*residual)(x);
  }

  out += res;

  return out.relu(true);
}

Sequential ResidualBlock::conv_dw(int64_t inp, int64_t oup, int64_t kernel, int64_t stride) {
  return sequential({
    conv2d(inp, inp, kernel, stride, (kernel - 1) / 2, 1, inp, false),
    batch_norm2d(inp, 1e-05, true),
    relu(true),
    conv2d(inp, oup, 1, 1, 0, 1, 1, false),
    batch_norm2d(oup, 1e-05, true),
    relu(true)
  });
}

MobileNetV2::MobileNetV2(
        int64_t n_class,
        bool useUpsample,
        bool useDeconvGroup,
        bool addEdge,
        float channelRatio,
        int64_t minChannel,
        bool video) {
  ARGUMENT_CHECK(!video, "video must be false");
  ARGUMENT_CHECK(!useDeconvGroup, "useDeconvGroup must be false");
  ARGUMENT_CHECK(!addEdge, "addEdge must be false");

  add_edge = addEdge;
  channel_ratio = channelRatio;
  min_channel = minChannel;
  use_deconv_group = useDeconvGroup;

  if (video) {
    ADD_SUB_MODULE(stage0, conv_bn, 4, depth(32), 3, 2);
  } else {
    ADD_SUB_MODULE(stage0, conv_bn, 3, depth(32), 3, 2);
  }

  ADD_SUB_MODULE(stage1, std::make_shared<InvertedResidual>, depth(32), depth(16), 1, 1);

  ADD_SUB_MODULE(stage2, sequential, {
    std::make_shared<InvertedResidual>(depth(16), depth(24), 2, 6),
    std::make_shared<InvertedResidual>(depth(24), depth(24), 1, 6),
  });

  ADD_SUB_MODULE(stage3, sequential, {
    std::make_shared<InvertedResidual>(depth(24), depth(32), 2, 6),
    std::make_shared<InvertedResidual>(depth(32), depth(32), 1, 6),
    std::make_shared<InvertedResidual>(depth(32), depth(32), 1, 6),
  });

  ADD_SUB_MODULE(stage4, sequential, {
    std::make_shared<InvertedResidual>(depth(32), depth(64), 2, 6),
    std::make_shared<InvertedResidual>(depth(64), depth(64), 1, 6),
    std::make_shared<InvertedResidual>(depth(64), depth(64), 1, 6),
    std::make_shared<InvertedResidual>(depth(64), depth(64), 1, 6),
  });

  ADD_SUB_MODULE(stage5, sequential, {
    std::make_shared<InvertedResidual>(depth(64), depth(96), 1, 6),
    std::make_shared<InvertedResidual>(depth(96), depth(96), 1, 6),
    std::make_shared<InvertedResidual>(depth(96), depth(96), 1, 6),
  });

  ADD_SUB_MODULE(stage6, sequential, {
    std::make_shared<InvertedResidual>(depth(96), depth(160), 2, 6),
    std::make_shared<InvertedResidual>(depth(160), depth(160), 1, 6),
    std::make_shared<InvertedResidual>(depth(160), depth(160), 1, 6),
  });

  ADD_SUB_MODULE(stage7, std::make_shared<InvertedResidual>, depth(160), depth(320), 1, 6);

  if (useUpsample) {
    ADD_SUB_MODULE(deconv1, upsample2d, 2, "bilinear");
    ADD_SUB_MODULE(deconv2, upsample2d, 2, "bilinear");
    ADD_SUB_MODULE(deconv3, upsample2d, 2, "bilinear");
    ADD_SUB_MODULE(deconv4, upsample2d, 2, "bilinear");
    ADD_SUB_MODULE(deconv5, upsample2d, 2, "bilinear");
  } else {
    ADD_SUB_MODULE(deconv1, conv_tranpose2d, depth(96), depth(96), 4, 2, 1, 0, false);
    ADD_SUB_MODULE(deconv2, conv_tranpose2d, depth(32), depth(32), 4, 2, 1, 0, false);
    ADD_SUB_MODULE(deconv3, conv_tranpose2d, depth(24), depth(24), 4, 2, 1, 0, false);
    ADD_SUB_MODULE(deconv4, conv_tranpose2d, depth(16), depth(16), 4, 2, 1, 0, false);
    ADD_SUB_MODULE(deconv5, conv_tranpose2d, depth(8), depth(8), 4, 2, 1, 0, false);
  }

  ADD_SUB_MODULE(transit1, std::make_shared<ResidualBlock>, depth(320), depth(96));
  ADD_SUB_MODULE(transit2, std::make_shared<ResidualBlock>, depth(96), depth(32));
  ADD_SUB_MODULE(transit3, std::make_shared<ResidualBlock>, depth(32), depth(24));
  ADD_SUB_MODULE(transit4, std::make_shared<ResidualBlock>, depth(24), depth(16));
  ADD_SUB_MODULE(transit5, std::make_shared<ResidualBlock>, depth(16), depth(8));

  ADD_SUB_MODULE(pred, conv2d, depth(8), n_class, 3, 1, 1, 1, 1, false);

  if (addEdge) {
    ADD_SUB_MODULE(edge, conv2d, depth(8), n_class, 3, 1, 1, 1, 1, false);
  }
}

int64_t MobileNetV2::depth(int64_t channels) {
  int64_t value = (std::min)(channels, min_channel);
  return (std::max)(value, int64_t(channels * channel_ratio));
}

Sequential MobileNetV2::conv_bn(int64_t inp, int64_t oup, int64_t kernel, int64_t stride) {
  return sequential({
    conv2d(inp, oup, kernel, stride, (kernel - 1) / 2, 1, 1, false),
    batch_norm2d(oup, 1e-05, true),
    relu(true),
  });
}

Tensor MobileNetV2::forward(Tensor x) {
  auto feature_1_2  = (*stage0)(x);
  feature_1_2  = (*stage1)(feature_1_2);

  auto feature_1_4  = (*stage2)(feature_1_2);
  auto feature_1_8  = (*stage3)(feature_1_4);
  auto feature_1_16 = (*stage4)(feature_1_8);
  feature_1_16 = (*stage5)(feature_1_16);

  auto feature_1_32 = (*stage6)(feature_1_16);
  feature_1_32 = (*stage7)(feature_1_32);

  auto up_1_16 = (*deconv1)((*transit1)(feature_1_32));
  auto up_1_8  = (*deconv2)((*transit2)(feature_1_16 + up_1_16));
  auto up_1_4  = (*deconv3)((*transit3)(feature_1_8 + up_1_8));
  auto up_1_2  = (*deconv4)((*transit4)(feature_1_4 + up_1_4));
  auto up_1_1  = (*deconv5)((*transit5)(up_1_2));

  auto pre = (*pred)(up_1_1);

  return pre;
}

}













