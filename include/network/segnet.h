#ifndef NN_SEGNET_H
#define NN_SEGNET_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/conv2d.h"
#include "module/instance_norm2d.h"
#include "module/batch_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::segnet {

class MakeDense : public ModuleImpl {
public:
  Conv2d conv;
  BatchNorm2d bn;
  Relu act;

public:
  MakeDense(int64_t n_channels, int64_t growth_rate);

  Tensor forward(Tensor) override;
};

class DenseBlock : public ModuleImpl {
public:
  Sequential dense_layers;

public:
  DenseBlock(int64_t n_channels, int64_t n_denselayer, int64_t growth_rate, bool reset_channel=false);

  Tensor forward(Tensor) override;
};

class ResidualDenseBlock : public ModuleImpl {
public:
  Conv2d conv;
  std::shared_ptr<DenseBlock> dense_block;
  BatchNorm2d bn;
  PRelu act;

  bool add;

public:
  ResidualDenseBlock(int64_t n_in, int64_t s, bool a);

  Tensor forward(Tensor) override;
};

class InputProjection : public ModuleImpl {
public:
  int64_t sampling_times;

public:
  InputProjection(int64_t s);

  Tensor forward(Tensor) override;
};

class ERDSegNet : public ModuleImpl {
public:
  std::shared_ptr<InputProjection> cascade1;
  std::shared_ptr<InputProjection> cascade2;
  std::shared_ptr<InputProjection> cascade3;
  std::shared_ptr<InputProjection> cascade4;

  Sequential head_conv;
  std::shared_ptr<ResidualDenseBlock> stage_0;

  Sequential ba_1;
  Sequential down_1;
  std::shared_ptr<ResidualDenseBlock> stage_1;

  Sequential ba_2;
  Sequential down_2;
  std::shared_ptr<ResidualDenseBlock> stage_2;

  Sequential ba_3;
  Sequential down_3;
  Sequential stage_3;

  Sequential ba_4;
  Sequential down_4;
  Sequential stage_4;

  Conv2d classifier;

  PRelu prelu;

  Sequential stage3_down;
  BatchNorm2d bn3;
  Conv2d conv_3;

  Sequential stage2_down;
  BatchNorm2d bn2;
  Conv2d conv_2;

  Sequential stage1_down;
  BatchNorm2d bn1;
  Conv2d conv_1;

  Sequential stage0_down;
  BatchNorm2d bn0;
  Conv2d conv_0;

public:
  ERDSegNet(int64_t classes=2);

  std::vector<Module> conv_bn_act(int64_t inp, int64_t oup, int64_t kernel_size=3, int64_t stride=1, int64_t padding=1);

  std::vector<Module> bn_act(int64_t inp);

  Tensor forward(Tensor) override;
};

class SegMattingNet : public ModuleImpl {
public:
  std::shared_ptr<ERDSegNet> seg_extract;

  Conv2d convF1;

  BatchNorm2d bn;

  Conv2d convF2;

public:
  SegMattingNet();

  Tensor forward(Tensor) override;
};

}

#endif