#include "common/tensor.h"
#include "module/conv2d.h"
#include "module/max_pooling2d.h"
#include "module/relu.h"
#include "module/sequential.h"
#include "module/sigmoid.h"
#include "module/upsample2d.h"
#include "network/unit.h"

namespace dlfm::nn::unit {

LayerNorm::LayerNorm(int64_t num_features, float eps, bool affine)
  : eps_(eps), affine_(affine), num_features_(num_features) {
  if (affine_) {
    gamma_ = Tensor::create({num_features}, ElementType::from<float>());
    beta_ = Tensor::create({num_features}, ElementType::from<float>());
  }
}

void LayerNorm::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  if (affine_) {
    std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

    if (parent_name_scope.empty()) {
      name_scope = torch_name_scope_;
    }

    gamma_.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "gamma" + TORCH_MODEL_FILE_SUFFIX);
    beta_.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "beta" + TORCH_MODEL_FILE_SUFFIX);
  }
}

Tensor LayerNorm::forward(Tensor x) {
  ARGUMENT_CHECK(4 == x.shape().ndims(), "shape dimension must be 4");

  auto mean = x.mean({1, 2, 3}, true);
  auto std = x.std({1, 2, 3}, true);
  std += eps_;

  x = (x - mean) / std;

  if (affine_) {
    x *= gamma_.reshape({1, num_features_, 1, 1});
    x += beta_.reshape({1, num_features_, 1, 1});
  }

  return x;
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
    ADD_SUB_MODULE(norm, instance_norm2d, output_dim);
  } else if ("none" != norm_type) {
    RUNTIME_ERROR("not supported norm_type:" << norm_type);
  }

  if ("relu" == activation_type) {
    ADD_SUB_MODULE(activation, relu, true);
  } else if ("none" != activation_type) {
    RUNTIME_ERROR("not supported activation_type:" << activation_type);
  }

  ADD_SUB_MODULE(conv, conv2d, input_dim, output_dim, kernel_size, stride, 0);
}

Tensor Conv2dBlock::forward(Tensor x) {
  auto y = (*conv)((*pad)(x));

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
}

Tensor ContentEncoder::forward(Tensor x ) {
  return (*model)(x);
}

// Decoder
Decoder::Decoder(int64_t n_upsample, int64_t n_res, int64_t dim, int64_t output_dim, 
  std::string res_norm, std::string activ, std::string  pad_type) {
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

}