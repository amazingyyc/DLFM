#ifndef NN_UNET_H
#define NN_UNET_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/conv_transpose2d.h"

namespace dlfm::nn {

class Down: public ModuleImpl {
public:
    Sequential op_;

    Down(int64_t in_channel, int64_t out_channel);

public:
  void torch_name_scope(std::string) override;

  std::vector<Module> sub_modules() override;

  Tensor forward(Tensor) override;
};

class Up: public ModuleImpl {
public:
  ConvTranpose2d up_;
  Sequential conv_;

  Up(int64_t in_channel, int64_t out_channel);

public:
  void torch_name_scope(std::string) override;

  std::vector<Module> sub_modules() override;

  Tensor forward(std::vector<Tensor>) override;
};

class UNet: public ModuleImpl {
public:
  Sequential input_;

  std::shared_ptr<Down> down1_;
  std::shared_ptr<Down> down2_;
  std::shared_ptr<Down> down3_;
  std::shared_ptr<Down> down4_;

  std::shared_ptr<Up> up1_;
  std::shared_ptr<Up> up2_;
  std::shared_ptr<Up> up3_;
  std::shared_ptr<Up> up4_;

  Sequential output_;

public:
  UNet(int64_t in_channels, int64_t out_channels);

  void torch_name_scope(std::string) override;

  std::vector<Module> sub_modules() override;

  Tensor forward(Tensor) override;
};



}

#endif //DLFM_UNET_H
