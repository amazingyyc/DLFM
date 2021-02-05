#include "network/slim.h"

namespace dlfm::nn::slim {

Sequential conv_bn(int64_t inp, int64_t oup, int64_t stride) {
  return sequential({
    conv2d(inp, oup, 3, stride, 1, 1, 1, false),
    batch_norm2d(oup),
    relu(true)
  });
}

Sequential depth_conv2d(int64_t inp, int64_t oup, int64_t kernel, int64_t stride, int64_t pad) {
  return sequential({
    conv2d(inp, inp, kernel, stride, pad, 1, inp, true),
    relu(true),
    conv2d(inp, oup, 1, 1, 0, 1, 1, true),
  });
}

Sequential conv_dw(int64_t inp, int64_t oup, int64_t stride, int64_t padding) {
  return sequential({
    conv2d(inp, inp, 3, stride, padding, 1, inp, false),
    batch_norm2d(inp),
    relu(true),

    conv2d(inp, oup, 1, 1, 0, 1, 1, false),
    batch_norm2d(oup),
    relu(true),
  });
}

Slim::Slim(): num_classes(2) {
  ADD_SUB_MODULE(conv1, conv_bn, 3, 16, 2);

  ADD_SUB_MODULE(conv2, conv_dw, 16, 32, 1);
  ADD_SUB_MODULE(conv3, conv_dw, 32, 32, 2);
  ADD_SUB_MODULE(conv4, conv_dw, 32, 32, 1);
  ADD_SUB_MODULE(conv5, conv_dw, 32, 64, 2);
  ADD_SUB_MODULE(conv6, conv_dw, 64, 64, 1);
  ADD_SUB_MODULE(conv7, conv_dw, 64, 64, 1);
  ADD_SUB_MODULE(conv8, conv_dw, 64, 64, 1);

  ADD_SUB_MODULE(conv9, conv_dw, 64, 128, 2);
  ADD_SUB_MODULE(conv10, conv_dw, 128, 128, 1);
  ADD_SUB_MODULE(conv11, conv_dw, 128, 128, 1);

  ADD_SUB_MODULE(conv12, conv_dw, 128, 256, 2);
  ADD_SUB_MODULE(conv13, conv_dw, 256, 256, 1);

  ADD_SUB_MODULE(fc, linear, 448, 143);
}

Tensor Slim::forward(Tensor inputs) {
  auto x1 = (*conv1)(inputs);
  auto x2 = (*conv2)(x1);
  auto x3 = (*conv3)(x2);
  auto x4 = (*conv4)(x3);
  auto x5 = (*conv5)(x4);
  auto x6 = (*conv6)(x5);
  auto x7 = (*conv7)(x6);
  auto x8 = (*conv8)(x7);

  auto output1 = x8;

  auto x9 = (*conv9)(x8);
  auto x10 = (*conv10)(x9);
  auto x11 = (*conv11)(x10);

  auto output2 = x11;

  auto x12 = (*conv12)(x11);
  auto x13 = (*conv13)(x12);

  auto output3 = x13;

  output1 = output1.mean(3).mean(2);
  output2 = output2.mean(3).mean(2);
  output3 = output3.mean(3).mean(2);

  auto output = (*fc)(output1.cat({output2, output3}, 1));

  return output;
}

}