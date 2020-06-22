#include "network/pfld.h"

namespace dlfm::nn::pfld {

Sequential conv_bn(int64_t inp, int64_t oup, int64_t kernel, int64_t stride, int64_t padding) {
  return sequential({
    conv2d(inp, oup, kernel, stride, padding, 1, false),
    batch_norm2d(oup),
    relu(true)});
}

Conv2d conv_1x1_bn(int64_t inp, int64_t oup) {
  return sequential({
    conv2d(inp, oup, 1, 1, 0, 1, false),
    batch_norm2d(oup),
    relu(true)});
}

InvertedResidual::InvertedResidual(int64_t inp, int64_t oup, int64_t s, bool use_res, int64_t expand_ratio) {
  stride = s;
  use_res_connect = use_res;

  ARGUMENT_CHECK(1 == stride || 2 == stride, "stride must be 1/2");

  ADD_SUB_MODULE(conv, sequential, {
    conv2d(inp, inp * expand_ratio, 1, 1, 0, 1, false),
    batch_norm2d(inp * expand_ratio),
    relu(true),
    Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, inp * expand_ratio, false),
    batch_norm2d(inp * expand_ratio),
    relu(true),
    Conv2d(inp * expand_ratio, oup, 1, 1, 0, 1, false),
    batch_norm2d(oup)});
}

Tensor InvertedResidual::forward(std::vector<Tensor> x) {
  if (use_res_connect) {
    auto y = (*conv)(x);
    y += x;

    return y;
  } else {
    return (*conv)(x);
  }
}

PFLDInference::PFLDInference() {
  ADD_SUB_MODULE(conv1, conv2d, 3, 64, 3, 2, 1, 1, false);
  ADD_SUB_MODULE(bn1, batch_norm2d, 64);

  ADD_SUB_MODULE(conv2, conv2d, 64, 64, 3, 1, 1, 1, false);
  ADD_SUB_MODULE(bn2, batch_norm2d, 64);

  ADD_SUB_MODULE(conv3_1, std::make_shared<InvertedResidual>, 64, 64, 2, false, 2);
  ADD_SUB_MODULE(block3_2, std::make_shared<InvertedResidual>, 64, 64, 1, true, 2);
  ADD_SUB_MODULE(block3_3, std::make_shared<InvertedResidual>, 64, 64, 1, true, 2);
  ADD_SUB_MODULE(block3_4, std::make_shared<InvertedResidual>, 64, 64, 1, true, 2);
  ADD_SUB_MODULE(block3_5, std::make_shared<InvertedResidual>, 64, 64, 1, true, 2);

  ADD_SUB_MODULE(conv4_1, std::make_shared<InvertedResidual>, 64, 128, 2, false, 2);

  ADD_SUB_MODULE(conv5_1, std::make_shared<InvertedResidual>, 128, 128, 1, false, 4);
  ADD_SUB_MODULE(conv5_2, std::make_shared<InvertedResidual>, 128, 128, 1, true, 4);
  ADD_SUB_MODULE(conv5_3, std::make_shared<InvertedResidual>, 128, 128, 1, true, 4);
  ADD_SUB_MODULE(conv5_4, std::make_shared<InvertedResidual>, 128, 128, 1, true, 4);
  ADD_SUB_MODULE(conv5_5, std::make_shared<InvertedResidual>, 128, 128, 1, true, 4);
  ADD_SUB_MODULE(conv5_6, std::make_shared<InvertedResidual>, 128, 128, 1, true, 4);

  ADD_SUB_MODULE(conv6_1, std::make_shared<InvertedResidual>, 128, 16, 1, false, 2);

  ADD_SUB_MODULE(conv7, conv_bn, 16, 32, 3, 2);

  ADD_SUB_MODULE(conv8, conv2d, 32, 128, 7, 1, 0);
  ADD_SUB_MODULE(bn8, batch_norm2d, 128);

  ADD_SUB_MODULE(fc, linear, 176, 196);
}

Tensor PFLDInference::forward(std::vector<Tensor> x) {
  x = ((*bn1)((*conv1)(x))).relu(true);
  x = ((*bn2)((*conv2)(x))).relu(true);

  x = (*conv3_1)(x);
  x = (*block3_2)(x);
  x = (*block3_3)(x);
  x = (*block3_4)(x);

  auto out1 = (*block3_5)(x);

  x = (*conv4_1)(out1);
  x = (*conv5_1)(x);
  x = (*block5_2)(x);
  x = (*block5_3)(x);
  x = (*block5_4)(x);
  x = (*block5_5)(x);
  x = (*block5_6)(x);
  x = (*conv6_1)(x);

  auto x1 = x.avg_pooling2d(14, 14);
  x1 = x1.view({x1.shape()[0], -1});

  x = (*conv7)(x);

  auto x2 = x.avg_pooling2d(7, 7);
  x2 = x2.view({ x2.shape()[0], -1 });

  auto x3 = ((*conv8)(x)).relu(true);
  x3 = x3.view({ x3.shape()[0], -1 });

  auto multi_scale = x1.cat(x2, 1).cat(x3, 1);

  auto landmarks = (*fc)(multi_scale);

  return landmarks;
}


}