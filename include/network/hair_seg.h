#ifndef NN_HAIR_SEG_H
#define NN_HAIR_SEG_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/prelu.h"
#include "module/conv2d.h"
#include "module/conv_transpose2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::hair_seg {

class DownBlock: public ModuleImpl {
public:
  Sequential blocks;

  PRelu active;

public:
  DownBlock(int64_t in_channel, int64_t mid_channel, int64_t out_channel);

  Tensor forward(Tensor) override;
};

class DownMaxPool2dBlock: public ModuleImpl {
public:
  int64_t in_channel;
  int64_t mid_channel;
  int64_t out_channel;

  Sequential blocks;

  Conv2d conv;
  PRelu active;

public:
  DownMaxPool2dBlock(int64_t in_channel, int64_t mid_channel, int64_t out_channel);

  std::vector<Tensor> compute(Tensor);
};

class UpBlock : public ModuleImpl {
public:
  Sequential blocks;

  Conv2d conv;
  PRelu active;

public:
  UpBlock(int64_t in_channel, int64_t mid_channel, int64_t out_channel);

  Tensor forward(Tensor) override;
};

class UpMaxPool2dBlock : public ModuleImpl {
public:
  Sequential blocks;
  Conv2d conv;
  PRelu active;

public:
  UpMaxPool2dBlock(int64_t in_channel, int64_t mid_channel, int64_t out_channel);

  Tensor forward(std::vector<Tensor>) override;
};

class ResNetBlockV1 : public ModuleImpl {
public:
  Sequential blocks;
  PRelu active;

public:
  ResNetBlockV1(int64_t in_channel, int64_t mid_channel);

  Tensor forward(Tensor) override;
};

class ResNetBlockV2 : public ModuleImpl {
public:
  Sequential blocks;
  PRelu active;

public:
  ResNetBlockV2(int64_t in_channel, int64_t mid_channel);

  Tensor forward(Tensor) override;
};

class HairSeg : public ModuleImpl {
public:
  Sequential input_block;

  std::shared_ptr<DownMaxPool2dBlock> down_block0;

  std::shared_ptr<DownBlock> down_block1;
  std::shared_ptr<DownBlock> down_block2;

  std::shared_ptr<DownMaxPool2dBlock> down_block3;

  std::shared_ptr<DownBlock> down_block4;
  std::shared_ptr<DownBlock> down_block5;
  std::shared_ptr<DownBlock> down_block6;
  std::shared_ptr<DownBlock> down_block7;

  std::shared_ptr<DownMaxPool2dBlock> down_block8;

  std::shared_ptr<DownBlock> down_block9;

  std::shared_ptr<ResNetBlockV1> resnet_block0;
  std::shared_ptr<ResNetBlockV2> resnet_block1;
  std::shared_ptr<ResNetBlockV1> resnet_block2;
  std::shared_ptr<DownBlock> resnet_block3;
  std::shared_ptr<ResNetBlockV1> resnet_block4;
  std::shared_ptr<DownBlock> resnet_block5;
  std::shared_ptr<ResNetBlockV1> resnet_block6;
  std::shared_ptr<ResNetBlockV2> resnet_block7;
  std::shared_ptr<ResNetBlockV1> resnet_block8;
  std::shared_ptr<DownBlock> resnet_block9;
  std::shared_ptr<ResNetBlockV1> resnet_block10;
  std::shared_ptr<ResNetBlockV1> resnet_block11;

  std::shared_ptr<UpMaxPool2dBlock> up_block0;
  std::shared_ptr<UpBlock> up_block1;
  std::shared_ptr<UpMaxPool2dBlock> up_block2;
  std::shared_ptr<UpBlock> up_block3;
  std::shared_ptr<UpMaxPool2dBlock> up_block4;
  std::shared_ptr<ResNetBlockV1> up_block5;

  Sequential output_block;

public:
  HairSeg(int64_t in_channel, int64_t out_channel);

  Tensor forward(Tensor) override;
};


}

#endif