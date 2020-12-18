#include "common/tensor.h"
#include "module/conv2d.h"
#include "module/relu.h"
#include "module/tanh.h"
#include "module/sequential.h"
#include "module/upsample2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "math/instance_norm2d.h"
#include "network/ugatit_tiny.h"

namespace dlfm::nn::ugatit_tiny {

DownBlock::DownBlock(int64_t in_channel, int64_t out_channel) {
  ADD_SUB_MODULE(blocks, sequential, {
    conv2d(in_channel, out_channel, 3, 2, 1, 1, 1, false),
    instance_norm2d(out_channel, 1e-05, false),
    relu(true)
  });
}

Tensor DownBlock::forward(Tensor x) {
  return (*blocks)(x);
}

ResnetBlock::ResnetBlock(int64_t dim) {
  ADD_SUB_MODULE(blocks, sequential, {
    conv2d(dim, dim, 3, 1, 1, 1, dim, false),
    instance_norm2d(dim, 1e-05, false),
    relu(true),
    conv2d(dim, dim, 1, 1, 0, 1, 1, false),
    instance_norm2d(dim, 1e-05, false),
  });
}

Tensor ResnetBlock::forward(Tensor x) {
  auto out = (*blocks)(x);
  out += x;
  return out;
}

AdaILN::AdaILN(int64_t num_features, float e) {
  eps = e;
  rho = Tensor::create({ 1, num_features, 1, 1 });
}

void AdaILN::load_torch_model(
  const std::unordered_map<std::string, Tensor> &tensor_map,
  std::string parent_name_scope) {
  DEF_ACTUALLY_TORCH_NAME_SCOPE;

  LOAD_TORCH_TENSOR(name_scope, "rho", rho, tensor_map);

  one_sub_rho = (1.0 - rho);
}

Tensor AdaILN::forward(std::vector<Tensor> input) {
  auto x     = input[0];
  auto gamma = input[1];
  auto beta  = input[2];

  auto b = x.shape()[0];
  auto c = x.shape()[1];
  auto h = x.shape()[2];
  auto w = x.shape()[3];

  auto out_in = x.reshape({ b * c, h * w }).norm2d(eps).reshape({ b, c, h, w });
  auto out_ln = x.reshape({ b, c* h * w }).norm2d(eps).reshape({ b, c, h, w });

  out_in *= rho;
  out_ln *= one_sub_rho;
  out_in += out_ln;

  out_in *= gamma.unsqueeze(2).unsqueeze(3);
  out_in += beta.unsqueeze(2).unsqueeze(3);

  return out_in;
}

ResnetAdaILNBlock::ResnetAdaILNBlock(int64_t dim) {
  ADD_SUB_MODULE(conv1, conv2d, dim, dim, 3, 1, 1, 1, dim, false);
  ADD_SUB_MODULE(norm1, std::make_shared<AdaILN>, dim);
  ADD_SUB_MODULE(relu1, relu, true);

  ADD_SUB_MODULE(conv2, conv2d, dim, dim, 1, 1, 0, 1, 1, false);
  ADD_SUB_MODULE(norm2, std::make_shared<AdaILN>, dim);
}

Tensor ResnetAdaILNBlock::forward(std::vector<Tensor> input) {
  auto x = input[0];
  auto gamma = input[1];
  auto beta = input[2];

  auto out = (*conv1)(x);
  out = (*norm1)({ out, gamma, beta });
  out = (*relu1)(out);
  out = (*conv2)(out);
  out = (*norm2)({ out, gamma, beta });

  out += x;

  return out;
}

UpBlock::UpBlock(int64_t in_channel, int64_t out_channel) {
  ADD_SUB_MODULE(blocks, sequential, {
    upsample2d(2, "bilinear"),
    conv2d(in_channel, out_channel, 3, 1, 1, 1, 1, false),
    instance_norm2d(out_channel, 1e-05, false),
    relu(true)
  });
}

Tensor UpBlock::forward(Tensor x) {
  return (*blocks)(x);
}

}