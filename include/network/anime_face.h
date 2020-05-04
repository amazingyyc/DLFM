#ifndef NN_ANIME_FACE_H
#define NN_ANIME_FACE_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/conv2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::anime_face {

class ResnetBlock: public ModuleImpl {
public:
  Sequential conv_block;

  ResnetBlock(int64_t dim, bool use_bias);

public:
  Tensor forward(Tensor) override;
};

class AdaILN: public ModuleImpl {
public:
  float eps;

  Tensor rho;

  AdaILN(int64_t num_features, float eps=1e-5);

public:
  Tensor forward(std::vector<Tensor>) override;
};

class ResnetAdaILNBlock: public ModuleImpl {
public:
  ReflectionPad2d pad1;
  Conv2d conv1;
  std::shared_ptr<AdaILN> norm1;
  Relu relu1;

  ReflectionPad2d pad2;
  Conv2d conv2;
  std::shared_ptr<AdaILN> norm2;

  ResnetAdaILNBlock(int64_t dim, bool use_bias);

public:
  Tensor forward(std::vector<Tensor>) override;
};

class ILN: public ModuleImpl {
public:
  float eps;

  Tensor rho;
  Tensor gamma;
  Tensor beta;

  ILN(int64_t num_features, float eps=1e-5);

public:
  Tensor forward(Tensor) override;
};

class AnimeFace: public ModuleImpl {
public:
  int64_t input_nc;
  int64_t output_nc;
  int64_t ngf;
  int64_t n_blocks;
  int64_t img_size;

  bool light;

  Sequential DownBlock;

  Linear gap_fc;
  Linear gmp_fc;

  Conv2d conv1x1;

  Relu relu;

  Sequential FC;

  Linear gamma;
  Linear beta;

  std::vector<std::shared_ptr<ResnetAdaILNBlock>> UpBlock1;

  Sequential UpBlock2;

  AnimeFace(
    int64_t input_nc,
    int64_t output_nc,
    int64_t ngf=64,
    int64_t n_blocks=6,
    int64_t img_size=256,
    bool light=false);

public:
  Tensor forward(Tensor) override;
};

}

#endif