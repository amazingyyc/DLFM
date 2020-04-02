#include "common/tensor.h"
#include "module/tiny_unet.h"
#include "module/max_pooling2d.h"
#include "module/conv2d.h"
#include "module/relu.h"
#include "module/sequential.h"
#include "module/sigmoid.h"

namespace dlfm::nn {

TinyDown::TinyDown(int64_t in_channel, int64_t out_channel) {
  ARGUMENT_CHECK(out_channel == 2 * in_channel, "UNet Down need out_channel == 2 * in_channel");

  auto max_pooling2d_op = max_pooling2d({2, 2}, {2, 2}, {0, 0});
  auto conv2d_op1 = conv2d(in_channel, out_channel, {3, 3}, {1, 1}, {1, 1});
  auto relu_op1   = relu(true);
  auto conv2d_op2 = conv2d(out_channel, out_channel, {3, 3}, {1, 1}, {1, 1});
  auto relu_op2   = relu(true);

  ADD_SUB_MODULE(op, sequential, { max_pooling2d_op, conv2d_op1, relu_op1, conv2d_op2, relu_op2 });
}

Tensor TinyDown::forward(Tensor input) {
  return (*op)(input);
}

TinyUp::TinyUp(int64_t in_channel, int64_t out_channel) {
  ARGUMENT_CHECK(in_channel == 2 * out_channel, "UNet up need in_channel == 2 * out_channel");

  ADD_SUB_MODULE(up, conv_tranpose2d, in_channel, out_channel, 3, 2, 1, 1);

  auto conv2d_op1 = conv2d(in_channel, out_channel, {3, 3}, {1, 1}, {1, 1});
  auto relu_op1   = relu(true);
  auto conv2d_op2 = conv2d(out_channel, out_channel, {3, 3}, {1, 1}, {1, 1});
  auto relu_op2   = relu(true);

  ADD_SUB_MODULE(conv, sequential, { conv2d_op1, relu_op1, conv2d_op2, relu_op2 });
}

Tensor TinyUp::forward(std::vector<Tensor> inputs) {
  auto x1 = inputs[0];
  auto x2 = inputs[1];

  x1 = (*up)(x1);

  ARGUMENT_CHECK(x2.shape()[3] >= x1.shape()[3], "shape is error");
  ARGUMENT_CHECK(x2.shape()[2] >= x1.shape()[2], "shape is error");

  if (x2.shape()[3] > x1.shape()[3] || x2.shape()[2] > x1.shape()[2]) {
    size_t diff_w = x2.shape()[3] - x1.shape()[3];
    size_t diff_h = x2.shape()[2] - x1.shape()[2];

    x1 = x1.pad({diff_w / 2, diff_w - diff_w / 2, diff_h / 2, diff_h - diff_h / 2});
  }

  auto x = x2.cat(x1, 1);

  return (*conv)(x);
}

TinyUNet::TinyUNet(int64_t in_channels, int64_t out_channels) {
  ADD_SUB_MODULE(input, sequential, { conv2d(in_channels, 32, 3, 1, 1), relu(true) });

  ADD_SUB_MODULE(down1, std::make_shared<TinyDown>, 32, 64);
  ADD_SUB_MODULE(down2, std::make_shared<TinyDown>, 64, 128);
  ADD_SUB_MODULE(down3, std::make_shared<TinyDown>, 128, 256);
  ADD_SUB_MODULE(down4, std::make_shared<TinyDown>, 256, 512);

  ADD_SUB_MODULE(up1, std::make_shared<TinyUp>, 512, 256);
  ADD_SUB_MODULE(up2, std::make_shared<TinyUp>, 256, 128);
  ADD_SUB_MODULE(up3, std::make_shared<TinyUp>, 128, 64);
  ADD_SUB_MODULE(up4, std::make_shared<TinyUp>, 64, 32);

  ADD_SUB_MODULE(output, sequential, { conv2d(32, out_channels, 1), sigmoid(true) });
}

Tensor TinyUNet::forward(Tensor x) {
  Tensor up4_out;
  {
    auto input_out = (*input)(x);

    Tensor up3_out;
    {
      auto down1_out = (*down1)(input_out);

      Tensor up2_out;
      {
        auto down2_out = (*down2)(down1_out);

        Tensor up1_out;
        {
          auto down3_out = (*down3)(down2_out);
          auto down4_out = (*down4)(down3_out);

          up1_out = (*up1)({down4_out, down3_out});
        }

        up2_out = (*up2)({up1_out, down2_out});
      }

      up3_out = (*up3)({up2_out, down1_out});
    }

    up4_out = (*up4)({up3_out, input_out});
  }

  return (*output)(up4_out);
}

}

