#ifndef NN_TINY_UNET_H
#define NN_TINY_UNET_H

#include "common/tensor.h"
#include "module/module.h"
#include "module/sequential.h"
#include "module/conv_transpose2d.h"

namespace dlfm::nn::tiny_unet {

class TinyDown : public ModuleImpl {
public:
  Sequential op;

  TinyDown(int64_t in_channel, int64_t out_channel);

public:
  Tensor forward(Tensor) override;
};

class TinyUp : public ModuleImpl {
public:
  ConvTranpose2d up;
  Sequential conv;

  TinyUp(int64_t in_channel, int64_t out_channel);

public:
  Tensor forward(std::vector <Tensor>) override;
};

class TinyUNet : public ModuleImpl {
public:
  Sequential input;

  std::shared_ptr <TinyDown> down1;
  std::shared_ptr <TinyDown> down2;
  std::shared_ptr <TinyDown> down3;
  std::shared_ptr <TinyDown> down4;

  std::shared_ptr <TinyUp> up1;
  std::shared_ptr <TinyUp> up2;
  std::shared_ptr <TinyUp> up3;
  std::shared_ptr <TinyUp> up4;

  Sequential output;

public:
  TinyUNet(int64_t in_channels, int64_t out_channels);

  Tensor forward(Tensor) override;
};

}

#endif //DLFM_TINY_UNET_H
