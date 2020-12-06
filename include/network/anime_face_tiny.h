#ifndef NN_ANIME_FACE_TINY_H
#define NN_ANIME_FACE_TINY_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/conv2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::anime_face_tiny {

class DownBlock: public ModuleImpl {
public:
  Sequential blocks;

public:
  DownBlock(int64_t in_channel, int64_t out_channel);

  Tensor forward(Tensor) override;
};

class ResnetBlock: public ModuleImpl {
public:
  Sequential blocks;

public:
  ResnetBlock(int64_t dim);

  Tensor forward(Tensor) override;
};

class AdaILN: public ModuleImpl {
public:
  float eps;

  Tensor rho;

public:
  AdaILN(int64_t num_features, float eps=1e-5);

  void load_torch_model(
    const std::unordered_map<std::string, Tensor> &tensor_map,
    std::string parent_name_scope) override;

  Tensor forward(std::vector<Tensor>) override;
};

class ResnetAdaILNBlock: public ModuleImpl {
public:
  Conv2d conv1;
  std::shared_ptr<AdaILN> norm1;
  Relu relu1;

  Conv2d conv2;
  std::shared_ptr<AdaILN> norm2;

public:
  ResnetAdaILNBlock(int64_t dim);

  Tensor forward(std::vector<Tensor>) override;
};

class UpBlock: public ModuleImpl {
public:
  Sequential blocks;

public:
  UpBlock(int64_t in_channel, int64_t out_channel);

  Tensor forward(Tensor) override;
};

class AnimeFaceTiny: public ModuleImpl {
public:
  int64_t n_blocks;

  Sequential DownBlock;

  Linear gap_fc;
  Linear gmp_fc;

  Conv2d conv1x1;

  Relu active;

  Sequential FC;

  Linear gamma;
  Linear beta;

  std::vector<std::shared_ptr<ResnetAdaILNBlock>> UpBlock1;

  Sequential UpBlock2;

public:
  AnimeFaceTiny(int64_t in_channel, int64_t out_channel, int64_t ngf=64, int64_t n_downsampling=2, int64_t n_blocks=4);

  Tensor forward(Tensor) override;
};

}

#endif
















