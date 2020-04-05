#ifndef NN_UNIT_H
#define NN_UNIT_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/conv2d.h"
#include "module/instance_norm2d.h"
#include "module/zero_pad2d.h"

namespace dlfm::nn::unit {

// LayerNorm
class LayerNorm: public ModuleImpl {
public:
  float eps_;
  bool affine_;
  int64_t num_features_;

  Tensor gamma_;
  Tensor beta_;

  LayerNorm(int64_t num_features, float eps=1e-5, bool affine=true);

public:
  Tensor forward(Tensor) override;
};

// Conv2dBlock
class Conv2dBlock: public ModuleImpl {
public:
  Module pad;
  Module norm;
  Module activation;
  Conv2d conv;

  Conv2dBlock(
    int64_t input_dim, 
    int64_t output_dim, 
    size_t kernel_size, 
    size_t stride,
    size_t padding = 0,
    std::string norm_type="none",
    std::string activation_type="relu", 
    std::string pad_type="zero");

public:
  Tensor forward(Tensor) override;
};

// ResBlock
class ResBlock: public ModuleImpl {
public:
  Sequential model;

  ResBlock(int64_t dim, std::string norm="in", std::string activation="relu", std::string pad_type="zero");

public:
  Tensor forward(Tensor) override;
};

// ResBlocks
class ResBlocks: public ModuleImpl {
public:
  Sequential model;

  ResBlocks(int64_t num_blocks,
            int64_t dim,
            std::string norm="in",
            std::string activation="relu",
            std::string pad_type="zero");

public:
  Tensor forward(Tensor) override;
};

// ContentEncoder
class ContentEncoder: public ModuleImpl {
public:
  Sequential model;

  ContentEncoder(
    int64_t n_downsample, 
    int64_t n_res, 
    int64_t input_dim, 
    int64_t dim, 
    std::string norm, 
    std::string activ, 
    std::string pad_type);

public:
  Tensor forward(Tensor) override;
};

// Decoder
class Decoder: public ModuleImpl {
public:
  Sequential model;

  Decoder(int64_t n_upsample, int64_t n_res, int64_t dim, int64_t output_dim, 
  std::string res_norm="adain", std::string activ="relu", std::string  pad_type="zero");

public:
  Tensor forward(Tensor) override;
};

}

#endif