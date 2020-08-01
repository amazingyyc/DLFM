#include "common/tensor.h"
#include "module/conv2d.h"
#include "module/relu.h"
#include "module/tanh.h"
#include "module/sequential.h"
#include "module/upsample2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "math/instance_norm2d.h"
#include "network/anime_face_tiny.h"

namespace dlfm::nn::anime_face_tiny {

DownBlock::DownBlock(int64_t in_channel, int64_t out_channel) {
  ADD_SUB_MODULE(blocks, sequential, {
    conv2d(in_channel, out_channel, 3, 2, 1, 1, false),
    instance_norm2d(out_channel, 1e-05, false),
    relu(true)
  });
}

Tensor DownBlock::forward(Tensor x) {
  return (*blocks)(x);
}

ResnetBlock::ResnetBlock(int64_t dim) {
  ADD_SUB_MODULE(blocks, sequential, {
    conv2d(dim, dim, 3, 1, 1, dim, false),
    instance_norm2d(dim, 1e-05, false),
    relu(true),
    conv2d(dim, dim, 1, 1, 0, 1, false),
    instance_norm2d(dim, 1e-05, false),
  });
}

Tensor ResnetBlock::forward(Tensor x) {
  return x + (*blocks)(x);
}

AdaILN::AdaILN(int64_t num_features, float e) {
  eps = e;

  rho = Tensor::create({1, num_features, 1, 1});
}

void AdaILN::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scope.empty()) {
    name_scope = torch_name_scope_;
  }

  rho.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "rho" + TORCH_MODEL_FILE_SUFFIX);
}

Tensor AdaILN::forward(std::vector<Tensor> x) {
  auto input = x[0];
  auto gamma = x[1];
  auto beta  = x[2];

  ARGUMENT_CHECK(4 == input.shape().ndims(), "input dimension must be 4");

  auto b = input.shape()[0];
  auto c = input.shape()[1];
  auto h = input.shape()[2];
  auto w = input.shape()[3];

  auto in_mean = input.mean({2, 3}, true);
  auto in_var = input.reshape({b, c, h * w}).var(-1, in_mean.reshape({b, c, 1})).reshape({b, c, 1, 1});
  in_var += eps;

  auto out_in = (input - in_mean) / in_var.sqrt(true);

  // ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
  auto ln_mean = input.mean({1, 2, 3}, true);
  auto ln_var = input.reshape({b, c * h * w}).var(-1, ln_mean.reshape({b, 1})).reshape({b, 1, 1, 1});
  ln_var += eps;

  auto out_ln = (input - ln_mean) / ln_var.sqrt(true);

  auto out = out_in * rho + out_ln * (1.0 - rho);

  out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3);

  return out;
}

ResnetAdaILNBlock::ResnetAdaILNBlock(int64_t dim) {
  ADD_SUB_MODULE(conv1, conv2d, dim, dim, 3, 1, 1, dim, false);
  ADD_SUB_MODULE(norm1, std::make_shared<AdaILN>, dim);
  ADD_SUB_MODULE(relu1, relu, true);

  ADD_SUB_MODULE(conv2, conv2d, dim, dim, 1, 1, 0, 1, false);
  ADD_SUB_MODULE(norm2, std::make_shared<AdaILN>, dim);
}

Tensor ResnetAdaILNBlock::forward(std::vector<Tensor> input) {
  auto x = input[0];
  auto gamma = input[1];
  auto beta = input[2];

  auto out = (*conv1)(x);
  out = (*norm1)({out, gamma, beta});
  out = (*relu1)(out);
  out = (*conv2)(out);
  out = (*norm2)({out, gamma, beta});

  return out + x;
}

UpBlock::UpBlock(int64_t in_channel, int64_t out_channel) {
  ADD_SUB_MODULE(blocks, sequential, {
    upsample2d(2, "bilinear"),
    conv2d(in_channel, out_channel, 3, 1, 1, 1, false),
    instance_norm2d(out_channel, 1e-05, false),
    relu(true)
  });
}

Tensor UpBlock::forward(Tensor x) {
  return (*blocks)(x);
}

AnimeFaceTiny::AnimeFaceTiny(int64_t in_channel, int64_t out_channel, int64_t ngf, int64_t n_downsampling, int64_t n) {
  n_blocks = n;

  std::vector<Module> DownBlockNodes;

  DownBlockNodes.emplace_back(conv2d(in_channel, ngf, 7, 1, 3, 1, false));
  DownBlockNodes.emplace_back(instance_norm2d(ngf, 1e-05, false));
  DownBlockNodes.emplace_back(std::make_shared<ReluImpl>(true));

  int64_t mult;

  for (int64_t i = 0; i < n_downsampling; ++i) {
    mult = (int64_t)pow(2, i);

    DownBlockNodes.emplace_back(std::make_shared<dlfm::nn::anime_face_tiny::DownBlock>(ngf * mult, ngf * mult * 2));
  }

  mult = (int64_t)pow(2, n_downsampling);

  for (int64_t i = 0; i < n_blocks; ++i) {
    DownBlockNodes.emplace_back(std::make_shared<ResnetBlock>(ngf * mult));
  }

  ADD_SUB_MODULE(DownBlock, sequential, DownBlockNodes);

  ADD_SUB_MODULE(gap_fc, linear, ngf * mult, 1, false);
  ADD_SUB_MODULE(gmp_fc, linear, ngf * mult, 1, false);
  ADD_SUB_MODULE(conv1x1, conv2d, ngf * mult * 2, ngf * mult, 1, 1, 0, 1, true);
  ADD_SUB_MODULE(active, std::make_shared<ReluImpl>, true);

  ADD_SUB_MODULE(FC, sequential, {
    linear(ngf * mult, ngf * mult, false),
    std::make_shared<ReluImpl>(true),
    linear(ngf * mult, ngf * mult, false),
    std::make_shared<ReluImpl>(true)
  });

  ADD_SUB_MODULE(gamma, linear, ngf * mult, ngf * mult, false);
  ADD_SUB_MODULE(beta, linear, ngf * mult, ngf * mult, false);

  for (int i = 0; i < n_blocks; ++i) {
    auto block = std::make_shared<ResnetAdaILNBlock>(ngf * mult);
    block->torch_name_scope("UpBlock1_" + std::to_string(i + 1));

    sub_modules_.emplace_back(block);
    UpBlock1.emplace_back(block);
  }

  std::vector<Module> UpBlock2Nodes;

  for (int64_t i = 0; i < n_downsampling; ++i) {
    mult = (int64_t)pow(2, (n_downsampling - i));

    UpBlock2Nodes.emplace_back(std::make_shared<UpBlock>(ngf * mult, ngf * mult / 2));
  }

  UpBlock2Nodes.emplace_back(conv2d(ngf, out_channel, 7, 1, 3, 1, false));
  UpBlock2Nodes.emplace_back(tanh(true));

  ADD_SUB_MODULE(UpBlock2, sequential, UpBlock2Nodes);
}

Tensor AnimeFaceTiny::forward(Tensor input) {
  // input must be [1, 3, 256, 256]
  // output [1, 3, 256, 256]
  ARGUMENT_CHECK(4 == input.ndims(), "input shape error");
  ARGUMENT_CHECK(input.shape() == Shape({1, 3, 256, 256}), "input shape error");

  auto x = (*DownBlock)(input);

  auto gap = x.adaptive_avg_pooling2d(1);
  // auto gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
  auto gap_weight = gap_fc->weight;
  gap = x * gap_weight.unsqueeze(2).unsqueeze(3);

  auto gmp = x.adaptive_max_pooling2d(1);
  // gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
  auto gmp_weight = gmp_fc->weight;
  gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3);

  x = gap.cat(gmp, 1);
  x = (*active)((*conv1x1)(x));

  auto x_ = x.adaptive_avg_pooling2d(1);
  x_ = (*FC)(x_.view({x_.shape()[0], -1}));

  auto ga = (*gamma)(x_);
  auto be = (*beta)(x_);

  for (auto& block : UpBlock1) {
    x = (*block)({x, ga, be});
  }

  auto out = (*UpBlock2)(x);

  return out;
}

}
















