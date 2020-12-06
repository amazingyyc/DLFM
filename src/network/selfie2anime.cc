#include "common/tensor.h"
#include "module/conv2d.h"
#include "module/max_pooling2d.h"
#include "module/relu.h"
#include "module/tanh.h"
#include "module/sequential.h"
#include "module/sigmoid.h"
#include "module/upsample2d.h"
#include "network/selfie2anime.h"

namespace dlfm::nn::selfie2anime {

LayerNorm::LayerNorm(int64_t num_features, float eps, bool affine)
  : eps_(eps), affine_(affine), num_features_(num_features) {
  if (affine_) {
    gamma_ = Tensor::create({num_features}, ElementType::from<float>());
    beta_ = Tensor::create({num_features}, ElementType::from<float>());
  }
}

void LayerNorm::load_torch_model(
    const std::unordered_map<std::string, Tensor> &tensor_map,
    std::string parent_name_scope) {
  if (affine_) {
    DEF_ACTUALLY_TORCH_NAME_SCOPE;

    LOAD_TORCH_TENSOR(name_scope, "gamma", gamma_, tensor_map);
    LOAD_TORCH_TENSOR(name_scope, "beta", beta_, tensor_map);
  }
}

Tensor LayerNorm::forward(Tensor x) {
  ARGUMENT_CHECK(4 == x.shape().ndims(), "shape dimension must be 4");

  auto b = x.shape()[0];
  auto c = x.shape()[1];
  auto h = x.shape()[2];
  auto w = x.shape()[3];

  auto mean = x.reshape({b, c * h * w}).mean(-1, true);
  auto std = x.reshape({ b, c * h * w }).std(-1, mean).reshape({ b, 1, 1, 1 });

  std += eps_;

  auto y = x - mean;
  y /= std;

  if (affine_) {
    y *= gamma_.reshape({ 1, c, 1, 1 });
    y += beta_.reshape({ 1, c, 1, 1 });
  }

  return y;
}

// Conv2dBlock
Conv2dBlock::Conv2dBlock(int64_t input_dim, int64_t output_dim,
                         size_t kernel_size, size_t stride, size_t padding,
                         std::string norm_type,
                         std::string activation_type,
                         std::string pad_type) {
  if ("zero" == pad_type) {
    ADD_SUB_MODULE(pad, zero_pad2d, padding);
  } else {
    RUNTIME_ERROR("not supported pad_type:" << pad_type);
  }

  if ("in" == norm_type) {
    ADD_SUB_MODULE(norm, instance_norm2d, output_dim, 1e-05, false);
  } else if ("ln" == norm_type) {
    ADD_SUB_MODULE(norm, std::make_shared<LayerNorm>, output_dim);
  } else {
    ARGUMENT_CHECK("none" == norm_type, "not supported norm_type:" << norm_type);
  }

  if ("relu" == activation_type) {
    ADD_SUB_MODULE(activation, relu, true);
  } else if ("tanh" == activation_type) {
    ADD_SUB_MODULE(activation, tanh, true);
  } else {
    ARGUMENT_CHECK("none" == activation_type, "not supported activation_type:" << activation_type);
  }

  ADD_SUB_MODULE(conv, conv2d, input_dim, output_dim, kernel_size, stride, 0);
}

Tensor Conv2dBlock::forward(Tensor x) {
  auto y = (*pad)(x);
  y = (*conv)(y);

  if (nullptr != norm) {
    y = (*norm)(y);
  }

  if (nullptr != activation) {
    y = (*activation)(y);
  }

  return y;
}

// ResBlock
ResBlock::ResBlock(int64_t dim, std::string norm, std::string activation, std::string pad_type) {
  ADD_SUB_MODULE(model, sequential, {
    std::make_shared<Conv2dBlock>(dim, dim, 3, 1, 1, norm, activation, pad_type),
    std::make_shared<Conv2dBlock>(dim, dim, 3, 1, 1, norm, "none", pad_type)
  });
}

Tensor ResBlock::forward(Tensor x) {
  auto out = (*model)(x);
  return (out += x);
}

// ResBlocks
ResBlocks::ResBlocks(int64_t num_blocks, int64_t dim, std::string norm, std::string activation, std::string pad_type) {
  std::vector<Module> blocks;

  for (int64_t i = 0; i < num_blocks; ++i) {
    blocks.emplace_back(std::make_shared<ResBlock>(dim, norm, activation, pad_type));
  }

  ADD_SUB_MODULE(model, sequential, blocks);
}

Tensor ResBlocks::forward(Tensor x) {
  return (*model)(x);
}

// ContentEncoder
ContentEncoder::ContentEncoder(
    int64_t n_downsample,
    int64_t n_res,
    int64_t input_dim,
    int64_t dim,
    std::string norm,
    std::string activ,
    std::string pad_type) {
  std::vector<Module> blocks;

  blocks.emplace_back(std::make_shared<Conv2dBlock>(input_dim, dim, 7, 1, 3, norm, activ, pad_type));

  for (int64_t i = 0; i < n_downsample; ++i) {
    blocks.emplace_back(std::make_shared<Conv2dBlock>(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type));
    dim *= 2;
  }

  blocks.emplace_back(std::make_shared<ResBlocks>(n_res, dim, norm, activ, pad_type));

  ADD_SUB_MODULE(model, sequential, blocks);

  output_dim = dim;
}

Tensor ContentEncoder::forward(Tensor x ) {
  return (*model)(x);
}

// Decoder
Decoder::Decoder(
  int64_t n_upsample,
  int64_t n_res,
  int64_t dim,
  int64_t output_dim,
  std::string res_norm,
  std::string activ,
  std::string  pad_type) {
  std::vector<Module> blocks;

  blocks.emplace_back(std::make_shared<ResBlocks>(n_res, dim, res_norm, activ, pad_type));

  for (int64_t i = 0; i < n_upsample; ++i) {
    blocks.emplace_back(upsample2d(2));
    blocks.emplace_back(std::make_shared<Conv2dBlock>(dim, dim / 2, 5, 1, 2, "ln", activ, pad_type));

    dim /= 2;
  }

  blocks.emplace_back(std::make_shared<Conv2dBlock>(dim, output_dim, 7, 1, 3, "none", "tanh", pad_type));

  ADD_SUB_MODULE(model, sequential, blocks);
}

Tensor Decoder::forward(Tensor x) {
  return (*model)(x);
}

VAEGen::VAEGen(
    int64_t input_dim,
    int64_t dim,
    int64_t n_downsample,
    int64_t n_res,
    std::string activ,
    std::string pad_type) {
  ADD_SUB_MODULE(enc, std::make_shared<ContentEncoder>, n_downsample, n_res, input_dim, dim, "in", activ, pad_type);
  ADD_SUB_MODULE(dec, std::make_shared<Decoder>, n_downsample, n_res, enc->output_dim, input_dim, "in", activ, pad_type);

  mean = Tensor::create({3});
  std = Tensor::create({3});

  float *mean_ptr = mean.data<float>();
  float *std_ptr = std.data<float>();

  mean_ptr[0] = 0.5;
  mean_ptr[1] = 0.5;
  mean_ptr[2] = 0.5;

  std_ptr[0] = 0.5;
  std_ptr[1] = 0.5;
  std_ptr[2] = 0.5;
}

Tensor VAEGen::forward(Tensor x) {
  // x is [h, w, 3] uint8 image
  ARGUMENT_CHECK(x.element_type().is<uint8_t>(), "VAEGen need uint8 input");
  ARGUMENT_CHECK(3 == x.shape().ndims() && 3 == x.shape()[2], "VAEGen input shape error");
  ARGUMENT_CHECK(x.shape()[0] >= 32 && x.shape()[1] >= 32,  "VAEGen input shape error");

  // -> [3, h, w]
  x = x.transpose({2, 0, 1});

  // float [3, h, w]
  x = x.cast(ElementType::from<float>());
  x *= (1 / 255.0);

  // normalize [-1, 1]
  x = x.normalize(mean, std, true);

  auto hidden = (*enc)(x.unsqueeze(0));

  auto y = (*dec)(hidden);

  // [3, h, w] (-1, 1)
  y = y.squeeze(0);
  y += 1.0;
  y *= (255.0 / 2.0);

  y = y.clamp(0, 255, true).cast(ElementType::from<uint8_t>());

  return y.transpose({1, 2, 0});
}


}