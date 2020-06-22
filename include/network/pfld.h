#ifndef NN_PFLD_H
#define NN_PFLD_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/conv2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::pfld {

Sequential conv_bn(int64_t inp, int64_t oup, int64_t kernel, int64_t stride, int64_t padding = 1);

Sequential conv_1x1_bn(int64_t inp, int64_t oup);

class InvertedResidual : public ModuleImpl {
public:
  int64_t stride;

  bool use_res_connect;

  Sequential conv;

  InvertedResidual(int64_t inp, int64_t oup, int64_t stride, bool use_res_connect, int64_t expand_ratio=6);

public:
  Tensor forward(std::vector<Tensor>) override;
};

class PFLDInference : public ModuleImpl {
public:
  Conv2d conv1;
  BatchNorm2d bn1;
  // Relu relu;

  Conv2d conv2;
  BatchNorm2d bn2;

  std::shared_ptr<InvertedResidual> conv3_1;
  std::shared_ptr<InvertedResidual> block3_2;
  std::shared_ptr<InvertedResidual> block3_3;
  std::shared_ptr<InvertedResidual> block3_4;
  std::shared_ptr<InvertedResidual> block3_5;

  std::shared_ptr<InvertedResidual> conv4_1;

  std::shared_ptr<InvertedResidual> conv5_1;
  std::shared_ptr<InvertedResidual> block5_2;
  std::shared_ptr<InvertedResidual> block5_3;
  std::shared_ptr<InvertedResidual> block5_4;
  std::shared_ptr<InvertedResidual> block5_5;
  std::shared_ptr<InvertedResidual> block5_6;

  std::shared_ptr<InvertedResidual> conv6_1;

  Sequential conv7;

  Conv2d conv8;
  BatchNorm2d bn8;

  Linear fc;

  PFLDInference();

public:
  Tensor forward(std::vector<Tensor>) override;
};

}