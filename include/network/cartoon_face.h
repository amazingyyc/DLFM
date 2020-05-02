#ifndef NN_CARTOON_FACE_H
#define NN_CARTOON_FACE_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/conv2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::cartoon_face {

class ConvBlock : public ModuleImpl {
public:
  int64_t dim_out;

  Sequential ConvBlock1;
  Sequential ConvBlock2;
  Sequential ConvBlock3;
  Sequential ConvBlock4;

  ConvBlock(int64_t dim_in, int64_t dim_out);

public:
  Tensor forward(Tensor) override;
};

class HourGlassBlock: public ModuleImpl {
public:
  std::shared_ptr<ConvBlock> ConvBlock1_1;
  std::shared_ptr<ConvBlock> ConvBlock1_2;
  std::shared_ptr<ConvBlock> ConvBlock2_1;
  std::shared_ptr<ConvBlock> ConvBlock2_2;
  std::shared_ptr<ConvBlock> ConvBlock3_1;
  std::shared_ptr<ConvBlock> ConvBlock3_2;
  std::shared_ptr<ConvBlock> ConvBlock4_1;
  std::shared_ptr<ConvBlock> ConvBlock4_2;

  std::shared_ptr<ConvBlock> ConvBlock5;

  std::shared_ptr<ConvBlock> ConvBlock6;
  std::shared_ptr<ConvBlock> ConvBlock7;
  std::shared_ptr<ConvBlock> ConvBlock8;
  std::shared_ptr<ConvBlock> ConvBlock9;

  HourGlassBlock(int64_t dim_in, int64_t dim_out);

public:
  Tensor forward(Tensor) override;
};

class ResnetBlock: public ModuleImpl {
public:
  Sequential conv_block;

  ResnetBlock(int64_t dim, bool use_bias=false);

public:
  Tensor forward(Tensor) override;
};

class HourGlass: public ModuleImpl {
public:
  bool use_res;

  Sequential HG;

  Conv2d Conv1;
  Conv2d Conv2;
  Conv2d Conv3;

  HourGlass(int64_t dim_in, int64_t dim_out, bool use_res=true);

public:
  Tensor forward(Tensor) override;
};

class AdaLIN: public ModuleImpl {
public:
  float eps;

  Tensor rho;

  AdaLIN(int64_t num_features, float eps=1e-5);

public:
  void load_torch_model(std::string model_folder, std::string parent_name_scope) override;

  Tensor forward(std::vector<Tensor>) override;
};

class SoftAdaLIN: public ModuleImpl {
public:
  std::shared_ptr<AdaLIN> norm;

  Sequential c_gamma;
  Sequential c_beta;

  Linear s_gamma;
  Linear s_beta;

  Tensor w_gamma;
  Tensor w_beta;

  SoftAdaLIN(int64_t num_features, float eps=1e-5);

public:
  void load_torch_model(std::string model_folder, std::string parent_name_scope) override;

  Tensor forward(std::vector<Tensor>) override;
};

class ResnetSoftAdaLINBlock: public ModuleImpl {
public:
  ReflectionPad2d pad1;
  Conv2d conv1;
  std::shared_ptr<SoftAdaLIN> norm1;
  Relu relu1;

  ReflectionPad2d pad2;
  Conv2d conv2;
  std::shared_ptr<SoftAdaLIN> norm2;

 ResnetSoftAdaLINBlock(int64_t dim, bool use_bias=false);

public:
  Tensor forward(std::vector<Tensor>) override;
};

class LIN: public ModuleImpl {
public:
  float eps;

  Tensor rho;
  Tensor gamma;
  Tensor beta;

  LIN(int64_t num_features, float eps=1e-5);

public:
  void load_torch_model(std::string model_folder, std::string parent_name_scope) override;

  Tensor forward(Tensor) override;
};

class CartoonFace: public ModuleImpl {
public:
  bool light;

  Sequential ConvBlock1;

  std::shared_ptr<HourGlass> HourGlass1;
  std::shared_ptr<HourGlass> HourGlass2;

  Sequential DownBlock1;
  Sequential DownBlock2;

  std::shared_ptr<ResnetBlock> EncodeBlock1;
  std::shared_ptr<ResnetBlock> EncodeBlock2;
  std::shared_ptr<ResnetBlock> EncodeBlock3;
  std::shared_ptr<ResnetBlock> EncodeBlock4;

  Linear gap_fc;
  Linear gmp_fc;

  Conv2d conv1x1;
  Relu relu;

  Sequential FC;

  std::shared_ptr<ResnetSoftAdaLINBlock> DecodeBlock1;
  std::shared_ptr<ResnetSoftAdaLINBlock> DecodeBlock2;
  std::shared_ptr<ResnetSoftAdaLINBlock> DecodeBlock3;
  std::shared_ptr<ResnetSoftAdaLINBlock> DecodeBlock4;

  Sequential UpBlock1;
  Sequential UpBlock2;

  std::shared_ptr<HourGlass> HourGlass3;
  std::shared_ptr<HourGlass> HourGlass4;

  Sequential ConvBlock2;

  CartoonFace(int64_t ngf=64, int64_t img_size=256, bool light=false);

public:
  Tensor forward(Tensor x) override;
};

}

#endif
