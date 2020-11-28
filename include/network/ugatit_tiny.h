#ifndef NN_UGATIT_TINY_H
#define NN_UGATIT_TINY_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/conv2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::ugatit_tiny {

class DownBlock: public ModuleImpl {
public:
  Sequential blocks;

public:
  DownBlock(int64_t in_channel, int64_t out_channel);

  Tensor forward(Tensor) override;
};

class ResnetBlock : public ModuleImpl {
public:
  Sequential blocks;

public:
  ResnetBlock(int64_t dim);

  Tensor forward(Tensor) override;
};

class AdaILN : public ModuleImpl {
public:
  float eps;

  Tensor rho;
  Tensor one_sub_rho;
public:
  AdaILN(int64_t num_features, float eps = 1e-5);

  void load_torch_model(std::string model_folder, std::string parent_name_scope) override;

  Tensor forward(std::vector<Tensor>) override;
};

class ResnetAdaILNBlock : public ModuleImpl {
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

class UpBlock : public ModuleImpl {
public:
  Sequential blocks;

public:
  UpBlock(int64_t in_channel, int64_t out_channel);

  Tensor forward(Tensor) override;
};

}