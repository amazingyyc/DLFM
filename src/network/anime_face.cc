#include "common/tensor.h"
#include "module/conv2d.h"
#include "module/relu.h"
#include "module/tanh.h"
#include "module/sequential.h"
#include "module/upsample2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "math/instance_norm2d.h"
#include "network/anime_face.h"

namespace dlfm::nn::anime_face {

ResnetBlock::ResnetBlock(int64_t dim, bool use_bias) {
  ADD_SUB_MODULE(conv_block, sequential, {
    reflection_pad2d(1),
    conv2d(dim, dim, 3, 1, 0, 1, use_bias),
    instance_norm2d(dim, 1e-05, false),
    relu(true),
    reflection_pad2d(1),
    conv2d(dim, dim, 3, 1, 0, 1, use_bias),
    instance_norm2d(dim, 1e-05, false)
  });
}

Tensor ResnetBlock::forward(Tensor x) {
  return x + (*conv_block)(x);
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

ResnetAdaILNBlock::ResnetAdaILNBlock(int64_t dim, bool use_bias) {
  ADD_SUB_MODULE(pad1, reflection_pad2d, 1);
  ADD_SUB_MODULE(conv1, conv2d, dim, dim, 3, 1, 0, 1, use_bias);
  ADD_SUB_MODULE(norm1, std::make_shared<AdaILN>, dim);
  ADD_SUB_MODULE(relu1, relu, true);

  ADD_SUB_MODULE(pad2, reflection_pad2d, 1);
  ADD_SUB_MODULE(conv2, conv2d, dim, dim, 3, 1, 0, 1, use_bias);
  ADD_SUB_MODULE(norm2, std::make_shared<AdaILN>, dim);
}

Tensor ResnetAdaILNBlock::forward(std::vector<Tensor> input) {
  auto x = input[0];
  auto gamma = input[1];
  auto beta = input[2];

  auto out = (*pad1)(x);
  out = (*conv1)(out);
  out = (*norm1)({out, gamma, beta});
  out = (*relu1)(out);
  out = (*pad2)(out);
  out = (*conv2)(out);
  out = (*norm2)({out, gamma, beta});

  return out + x;
}

ILN::ILN(int64_t num_features, float e) {
  eps = e;

  rho = Tensor::create({1, num_features, 1, 1});
  gamma = Tensor::create({1, num_features, 1, 1});
  beta = Tensor::create({1, num_features, 1, 1});
}

void ILN::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scope.empty()) {
    name_scope = torch_name_scope_;
  }

  rho.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "rho" + TORCH_MODEL_FILE_SUFFIX);
  gamma.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "gamma" + TORCH_MODEL_FILE_SUFFIX);
  beta.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "beta" + TORCH_MODEL_FILE_SUFFIX);
}

Tensor ILN::forward(Tensor input) {
  ARGUMENT_CHECK(4 == input.ndims(), "input ndims must be 4");

  auto b = input.shape()[0];
  auto c = input.shape()[1];
  auto h = input.shape()[2];
  auto w = input.shape()[3];

  auto in_mean = input.mean({2, 3}, true);
  auto in_var = input.reshape({b, c , h * w}).var(-1, in_mean.reshape({b, c, 1})).reshape({b, c, 1, 1});
  in_var += eps;

  auto out_in = (input - in_mean) / (in_var).sqrt(true);

  auto ln_mean = input.mean({1, 2, 3}, true);
  auto ln_var = input.reshape({b, c * h * w}).var(-1, ln_mean.reshape({b, 1})).reshape({b, 1, 1, 1});
  ln_var += eps;

  auto out_ln = (input - ln_mean) / ln_var.sqrt(true);

  auto out = rho * out_in + (1.0 - rho) * out_ln;
  out = out * gamma + beta;

  return out;
}

AnimeFace::AnimeFace(
  int64_t i_nc,
  int64_t o_nc,
  int64_t n,
  int64_t n_b,
  int64_t i_size,
  bool l) {
  input_nc = i_nc;
  output_nc = o_nc;
  ngf = n;
  n_blocks = n_b;
  img_size = i_size;
  light = l;

  std::vector<Module> DownBlockNodes;

  DownBlockNodes.emplace_back(reflection_pad2d(3));
  DownBlockNodes.emplace_back(conv2d(input_nc, ngf, 7, 1, 0, 1, false));
  DownBlockNodes.emplace_back(instance_norm2d(ngf, 1e-05, false));
  DownBlockNodes.emplace_back(std::make_shared<ReluImpl>(true));

  int64_t mult = 1;
  for (int i = 0; i < 2; ++i) {
    DownBlockNodes.emplace_back(reflection_pad2d(1));
    DownBlockNodes.emplace_back(conv2d(ngf * mult, ngf * mult * 2, 3, 2, 0, 1, false));
    DownBlockNodes.emplace_back(instance_norm2d(ngf * mult * 2, 1e-05, false));
    DownBlockNodes.emplace_back(std::make_shared<ReluImpl>(true));

    mult *= 2;
  }

  mult = 4;

  for (int i = 0; i < n_blocks; ++i) {
    DownBlockNodes.emplace_back(std::make_shared<ResnetBlock>(ngf * mult, false));
  }

  ADD_SUB_MODULE(DownBlock, sequential, DownBlockNodes);

  ADD_SUB_MODULE(gap_fc, linear, ngf * mult, 1, false);
  ADD_SUB_MODULE(gmp_fc, linear, ngf * mult, 1, false);
  ADD_SUB_MODULE(conv1x1, conv2d, ngf * mult * 2, ngf * mult, 1, 1, 0, 1, true);
  ADD_SUB_MODULE(relu, std::make_shared<ReluImpl>, true);

  if (light) {
    ADD_SUB_MODULE(FC, sequential, {
      linear(ngf * mult, ngf * mult, false),
      std::make_shared<ReluImpl>(true),
      linear(ngf * mult, ngf * mult, false),
      std::make_shared<ReluImpl>(true)
    });
  } else {
    ADD_SUB_MODULE(FC, sequential, {
      linear(img_size / mult * img_size / mult * ngf * mult, ngf * mult, false),
      std::make_shared<ReluImpl>(true),
      linear(ngf * mult, ngf * mult, false),
      std::make_shared<ReluImpl>(true)
    });
  }

  ADD_SUB_MODULE(gamma, linear, ngf * mult, ngf * mult, false);
  ADD_SUB_MODULE(beta, linear, ngf * mult, ngf * mult, false);

  for (int i = 0; i < n_blocks; ++i) {
    auto block = std::make_shared<ResnetAdaILNBlock>(ngf * mult, false);
    block->torch_name_scope("UpBlock1_" + std::to_string(i + 1));

    sub_modules_.emplace_back(block);
    UpBlock1.emplace_back(block);
  }

  std::vector<Module> UpBlock2Nodes;

  mult = 4;

  for (int i = 0; i < 2; ++i) {
    UpBlock2Nodes.emplace_back(upsample2d(2, "nearest"));
    UpBlock2Nodes.emplace_back(reflection_pad2d(1));
    UpBlock2Nodes.emplace_back(conv2d(ngf * mult, ngf * mult / 2, 3, 1, 0, 1, false));
    UpBlock2Nodes.emplace_back(std::make_shared<ILN>(ngf * mult / 2));
    UpBlock2Nodes.emplace_back(std::make_shared<ReluImpl>(true));

    mult /= 2;
  }

  UpBlock2Nodes.emplace_back(reflection_pad2d(3));
  UpBlock2Nodes.emplace_back(conv2d(ngf, output_nc, 7, 1, 0, 1, false));
  UpBlock2Nodes.emplace_back(tanh(true));

  ADD_SUB_MODULE(UpBlock2, sequential, UpBlock2Nodes);
}

Tensor AnimeFace::forward(Tensor input) {
  auto x = (*DownBlock)(input);

  auto gap = x.adaptive_avg_pooling2d(1);
  //auto gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
  auto gap_weight = gap_fc->weight;
  gap = x * gap_weight.unsqueeze(2).unsqueeze(3);

  auto gmp = x.adaptive_max_pooling2d(1);
  //gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
  auto gmp_weight = gmp_fc->weight;
  gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3);

  x = gap.cat(gmp, 1);
  x = (*relu)((*conv1x1)(x));

  Tensor x_;

  if (light) {
    x_ = x.adaptive_avg_pooling2d(1);
    x_ = (*FC)(x_.view({x_.shape()[0], -1}));
  } else {
    x_ = (*FC)(x.view({x.shape()[0], -1}));
  }

  auto ga = (*gamma)(x_);
  auto be = (*beta)(x_);

  for (auto block : UpBlock1) {
    x = (*block)({x, ga, be});
  }

  auto out = (*UpBlock2)(x);

  return out;
}

}