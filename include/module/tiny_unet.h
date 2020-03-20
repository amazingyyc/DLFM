#ifndef NN_TINY_UNET_H
#define NN_TINY_UNET_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/conv_transpose2d.h"

namespace dlfm::nn {

class TinyDown : public ModuleImpl {
public:
  Sequential op_;

  TinyDown(int64_t in_channel, int64_t out_channel);

public:
  void torch_name_scope(std::string) override;

  std::vector <Module> sub_modules() override;

  Tensor forward(Tensor) override;
};

class TinyUp : public ModuleImpl {
public:
  ConvTranpose2d up_;
  Sequential conv_;

  TinyUp(int64_t in_channel, int64_t out_channel);

public:
  void torch_name_scope(std::string) override;

  std::vector <Module> sub_modules() override;

  Tensor forward(std::vector <Tensor>) override;
};

class TinyUNet : public ModuleImpl {
public:
  Sequential input_;

  std::shared_ptr <TinyDown> down1_;
  std::shared_ptr <TinyDown> down2_;
  std::shared_ptr <TinyDown> down3_;
  std::shared_ptr <TinyDown> down4_;

  std::shared_ptr <TinyUp> up1_;
  std::shared_ptr <TinyUp> up2_;
  std::shared_ptr <TinyUp> up3_;
  std::shared_ptr <TinyUp> up4_;

  Sequential output_;

public:
  TinyUNet(int64_t in_channels, int64_t out_channels);

  void torch_name_scope(std::string) override;

  std::vector <Module> sub_modules() override;

  Tensor forward(Tensor) override;
};

}

#endif //DLFM_TINY_UNET_H
