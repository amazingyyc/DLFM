#include "common/tensor.h"
#include "module/unet.h"
#include "module/max_pooling2d.h"
#include "module/conv2d.h"
#include "module/relu.h"
#include "module/sequential.h"
#include "module/sigmoid.h"

namespace dlfm::nn {

Down::Down(int64_t in_channel, int64_t out_channel) {
    ARGUMENT_CHECK(out_channel == 2 * in_channel, "UNet Down need out_channel == 2 * in_channel");

    auto max_pooling2d_op = max_pooling2d({2, 2}, {2, 2}, {0, 0});
    auto conv2d_op1 = conv2d(in_channel, out_channel, {5, 5}, {1, 1}, {2, 2});
    auto relu_op1 = relu(true);
    auto conv2d_op2 = conv2d(out_channel, out_channel, {5, 5}, {1, 1}, {2, 2});
    auto relu_op2 = relu(true);

    op_ = sequential({max_pooling2d_op, conv2d_op1, relu_op1, conv2d_op2, relu_op2});
}

void Down::torch_name_scope(std::string name) {
  torch_name_scope_ = name;

  op_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "op");
}

std::vector<Module> Down::sub_modules() {
  return { op_ };
}

Tensor Down::forward(Tensor input) {
    return (*op_)(input);
}

Up::Up(int64_t in_channel, int64_t out_channel) {
    ARGUMENT_CHECK(in_channel == 2 * out_channel, "UNet up need in_channel == 2 * out_channel");

    up_ = conv_tranpose2d(in_channel, out_channel, {5, 5}, {2, 2}, {2, 2}, {1, 1});

    auto conv2d_op1 = conv2d(in_channel, out_channel, {5, 5}, {1, 1}, {2, 2});
    auto relu_op1 = relu(true);
    auto conv2d_op2 = conv2d(out_channel, out_channel, {5, 5}, {1, 1}, {2, 2});
    auto relu_op2 = relu(true);

    conv_ = sequential({conv2d_op1,
                        relu_op1,
                        conv2d_op2,
                        relu_op2});
}

void Up::torch_name_scope(std::string name) {
  torch_name_scope_ = name;

  up_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "up");
  conv_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv");
}

std::vector<Module> Up::sub_modules() {
  return { up_, conv_ };
}

Tensor Up::forward(std::vector<Tensor> inputs) {
    auto x1 = inputs[0];
    auto x2 = inputs[1];

    x1 = (*up_)(x1);

    ARGUMENT_CHECK(x2.shape()[3] >= x1.shape()[3], "shape is error");
    ARGUMENT_CHECK(x2.shape()[2] >= x1.shape()[2], "shape is error");

    if (x2.shape()[3] > x1.shape()[3] || x2.shape()[2] > x1.shape()[2]) {
      size_t diff_w = x2.shape()[3] - x1.shape()[3];
      size_t diff_h = x2.shape()[2] - x1.shape()[2];

      x1 = x1.pad({diff_w/2, diff_w - diff_w/2, diff_h/2, diff_h - diff_h/2});
    }

    auto x = x2.cat(x1, 1);

    return (*conv_)(x);
}

UNet::UNet(int64_t in_channels, int64_t out_channels) {
  input_ = sequential({
    conv2d(in_channels, 32, {5, 5}, {1, 1}, {2, 2}),
    relu(true)
  });

  down1_ = std::make_shared<Down>(32, 64);
  down2_ = std::make_shared<Down>(64, 128);
  down3_ = std::make_shared<Down>(128, 256);
  down4_ = std::make_shared<Down>(256, 512);

  up1_  = std::make_shared<Up>(512, 256);
  up2_  = std::make_shared<Up>(256, 128);
  up3_  = std::make_shared<Up>(128, 64);
  up4_  = std::make_shared<Up>(64, 32);

  output_ = sequential({
    conv2d(32, out_channels, {1, 1}, {1, 1}, {0, 0}),
    sigmoid(true)
  });
}

void UNet::torch_name_scope(std::string name) {
  torch_name_scope_ = name;

  input_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "input");

  down1_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "down1");
  down2_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "down2");
  down3_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "down3");
  down4_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "down4");

  up1_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "up1");
  up2_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "up2");
  up3_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "up3");
  up4_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "up4");

  output_->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "output");
}

std::vector<Module> UNet::sub_modules() {
  return { input_ , down1_ , down2_ , down3_ , down4_ , up1_ , up2_ , up3_ , up4_ , output_ };
}

Tensor UNet::forward(Tensor x) {
  Tensor up4_out;
  {
      auto input_out = (*input_)(x);

      Tensor up3_out;
      {
          auto down1_out = (*down1_)(input_out);

          Tensor up2_out;
          {
              auto down2_out = (*down2_)(down1_out);

              Tensor up1_out;
              {
                  auto down3_out = (*down3_)(down2_out);
                  auto down4_out = (*down4_)(down3_out);

                  up1_out = (*up1_)({down4_out, down3_out});
              }

              up2_out = (*up2_)({up1_out, down2_out});
          }

          up3_out = (*up3_)({up2_out, down1_out});
      }

      up4_out = (*up4_)({up3_out, input_out});
  }

  return (*output_)(up4_out);
}


}