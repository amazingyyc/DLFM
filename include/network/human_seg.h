#ifndef NN_HUMAN_SEG_H
#define NN_HUMAN_SEG_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/conv2d.h"
#include "module/conv_transpose2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::human_seg {

class InvertedResidual: public ModuleImpl {
public:
  int64_t stride;

  bool use_res_connect;

  Sequential conv;

  InvertedResidual(int64_t inp, int64_t oup, int64_t stride, int64_t expansion, int64_t dilation=1);

public:
  Tensor forward(Tensor) override;
};

class MobileNetV2: public ModuleImpl {
public:
  int64_t num_classes;

  int64_t last_channel;

  Sequential features;
  Sequential classifier;

  explicit MobileNetV2(float alpha=1.0, int64_t expansion=6, int64_t num_classes=1000);

  int64_t make_divisible(float v, int64_t divisor, int64_t min_value=-1);

  Sequential conv_bn(int64_t, int64_t , int64_t );

  Sequential conv_1x1_bn(int64_t inp, int64_t oup);

public:
  Tensor forward(Tensor) override;
};

class DecoderBlock: public ModuleImpl {
public:
  ConvTranpose2d deconv;
  std::shared_ptr<InvertedResidual> block_unit;

  DecoderBlock(int64_t in_channels, int64_t out_channels, std::shared_ptr<InvertedResidual> block_unit);

public:
  Tensor forward(std::vector<Tensor>) override;
};

class HumanSeg: public ModuleImpl {
public:
  Tensor mean;
  Tensor std;

  std::shared_ptr<MobileNetV2> backbone;

  std::shared_ptr<DecoderBlock> decoder1;
  std::shared_ptr<DecoderBlock> decoder2;
  std::shared_ptr<DecoderBlock> decoder3;
  std::shared_ptr<DecoderBlock> decoder4;

  Sequential conv_last;

  explicit HumanSeg(int64_t num_classes=2);

public:
  Tensor forward(Tensor) override;
};

}

#endif











