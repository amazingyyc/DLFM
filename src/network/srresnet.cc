#include "common/tensor.h"
#include "module/conv2d.h"
#include "module/relu6.h"
#include "module/tanh.h"
#include "module/sequential.h"
#include "network/srresnet.h"

namespace dlfm::nn::srresnet {

ResNet::ResNet(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding) {
  ADD_SUB_MODULE(conv1, conv2d, in_channels, out_channels, kernel_size, stride, padding);
  ADD_SUB_MODULE(relu1, relu6, true);
  ADD_SUB_MODULE(conv2, conv2d, in_channels, out_channels, kernel_size, stride, padding);
  ADD_SUB_MODULE(relu2, relu6, true);
}

Tensor ResNet::forward(Tensor x) {
  auto res = x;

  x = (*conv1)(x);
  x = (*relu1)(x);
  x = (*conv2)(x);

  return (*relu2)(x + res);
}

SRResNet::SRResNet(int64_t in_channels, int64_t out_channels) {
  ADD_SUB_MODULE(input, sequential, {
    conv2d(in_channels, 32, 3, 1, 1),
    relu6(true)
  });

  ADD_SUB_MODULE(res1, std::make_shared<ResNet>, 32, 32);
  ADD_SUB_MODULE(res2, std::make_shared<ResNet>, 32, 32);

  ADD_SUB_MODULE(output, sequential, {
    conv2d(32, out_channels, 3, 1, 1),
    tanh(true)
  });
}

Tensor SRResNet::forward(Tensor x) {
  x = (*input)(x);
  x = (*res1)(x);
  x = (*res2)(x);
  x = x.upsample2d(2, "bilinear");
  x = (*output)(x);

  return x;
}

}