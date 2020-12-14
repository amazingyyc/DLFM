#include "module/conv2d.h"
#include "module/prelu.h"
#include "module/sequential.h"
#include "module/reflection_pad2d.h"
#include "module/conv_transpose2d.h"
#include "network/hair_seg.h"

namespace dlfm::nn::hair_seg {

DownBlock::DownBlock(int64_t in_channel, int64_t mid_channel, int64_t out_channel) {
  ADD_SUB_MODULE(blocks, sequential, {
    conv2d(in_channel, mid_channel, 1, 1, 0, 1, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, mid_channel, 3, 1, 1, mid_channel, true),
    conv2d(mid_channel, mid_channel, 1, 1, 0, 1, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, mid_channel, 3, 1, 1, mid_channel, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, out_channel, 1, 1, 0, 1, true),
  });

  ADD_SUB_MODULE(active, prelu, true, out_channel);
}

Tensor DownBlock::forward(Tensor x) {
  auto y = (*blocks)(x);
  y += x;

  return (*active)(y);
}

DownMaxPool2dBlock::DownMaxPool2dBlock(int64_t in, int64_t mid, int64_t out)
  :in_channel(in), mid_channel(mid), out_channel(out) {
  ADD_SUB_MODULE(blocks, sequential, {
    conv2d(in_channel, mid_channel, 2, 2, 0, 1, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, mid_channel, 3, 1, 1, mid_channel, true),
    conv2d(mid_channel, mid_channel, 1, 1, 0, 1, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, mid_channel, 3, 1, 1, mid_channel, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, out_channel, 1, 1, 0, 1, true),
  });

  if (in_channel != out_channel) {
    ADD_SUB_MODULE(conv, conv2d, in_channel, out_channel, 1, 1, 0, 1, true);
  }

  ADD_SUB_MODULE(active, prelu, true, out_channel);
}

std::vector<Tensor> DownMaxPool2dBlock::compute(Tensor x) {
  auto y = (*blocks)(x);

  auto max_out = x.max_pooling2d_with_indices({2, 2}, {2, 2}, {0, 0});

  auto max_pool2d_x = max_out[0];
  auto max_pool2d_x_indices = max_out[1];

  if (in_channel != out_channel) {
    max_pool2d_x = (*conv)(max_pool2d_x);
  }

  y += max_pool2d_x;
  y = (*active)(y);

  return { y, max_pool2d_x_indices };
}

UpBlock::UpBlock(int64_t in_channel, int64_t mid_channel, int64_t out_channel) {
  ADD_SUB_MODULE(blocks, sequential, {
    conv2d(in_channel, mid_channel, 1, 1, 0, 1, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, mid_channel, 3, 1, 1, 1, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, out_channel, 1, 1, 0, 1, true),
  });

  ADD_SUB_MODULE(conv, conv2d, in_channel, out_channel, 1, 1, 0, 1, true);
  ADD_SUB_MODULE(active, prelu, true, out_channel);
}

Tensor UpBlock::forward(Tensor x) {
  auto y = (*blocks)(x);
  y += (*conv)(x);

  return (*active)(y);
}

UpMaxPool2dBlock::UpMaxPool2dBlock(int64_t in_channel, int64_t mid_channel, int64_t out_channel) {
  ADD_SUB_MODULE(blocks, sequential, {
    conv2d(in_channel, mid_channel, 1, 1, 0, 1, true),
    prelu(true, mid_channel),
    conv_tranpose2d(mid_channel, mid_channel, 3, 2, 1, 1, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, out_channel, 1, 1, 0, 1, true),
  });

  ADD_SUB_MODULE(conv, conv2d, in_channel, out_channel, 1, 1, 0, 1, true);
  ADD_SUB_MODULE(active, prelu, true, out_channel);
}

Tensor UpMaxPool2dBlock::forward(std::vector<Tensor> inputs) {
  auto x = inputs[0];
  auto indices = inputs[1];

  auto y = (*blocks)(x);
  auto up_x = (*conv)(x);
  up_x = up_x.max_unpooling2d(indices, {2, 2}, {2, 2}, {0, 0});

  y += up_x;

  return (*active)(y);
}

ResNetBlockV1::ResNetBlockV1(int64_t in_channel, int64_t mid_channel) {
  ADD_SUB_MODULE(blocks, sequential, {
    conv2d(in_channel, mid_channel, 1, 1, 0, 1, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, mid_channel, 3, 1, 1, 1, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, in_channel, 1, 1, 0, 1, true),
  });

  ADD_SUB_MODULE(active, prelu, true, in_channel);
}

Tensor ResNetBlockV1::forward(Tensor x) {
  auto y = (*blocks)(x);
  y += x;

  return (*active)(y);
}

ResNetBlockV2::ResNetBlockV2(int64_t in_channel, int64_t mid_channel) {
  ADD_SUB_MODULE(blocks, sequential, {
    conv2d(in_channel, mid_channel, 1, 1, 0, 1, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, mid_channel, 5, 1, 2, mid_channel, true),
    conv2d(mid_channel, mid_channel, 1, 1, 0, 1, true),
    prelu(true, mid_channel),
    conv2d(mid_channel, in_channel, 1, 1, 0, 1, true),
  });

  ADD_SUB_MODULE(active, prelu, true, in_channel);
}

Tensor ResNetBlockV2::forward(Tensor x) {
  auto y = (*blocks)(x);
  y += x;

  return (*active)(y);
}

HairSeg::HairSeg(int64_t in_channel, int64_t out_channel) {
  ADD_SUB_MODULE(input_block, sequential, {
    conv2d(in_channel, 8, 2, 2, 0, 1, true),
    prelu(true, 8),
    conv2d(8, 32, 2, 2, 0, 1, true),
    prelu(true, 32),
  });

  ADD_SUB_MODULE(down_block0, std::make_shared<DownMaxPool2dBlock>, 32, 16, 64);
  ADD_SUB_MODULE(down_block1, std::make_shared<DownBlock>, 64, 16, 64);
  ADD_SUB_MODULE(down_block2, std::make_shared<DownBlock>, 64, 16, 64);
  ADD_SUB_MODULE(down_block3, std::make_shared<DownMaxPool2dBlock>, 64, 32, 128);
  ADD_SUB_MODULE(down_block4, std::make_shared<DownBlock>, 128, 16, 128);
  ADD_SUB_MODULE(down_block5, std::make_shared<DownBlock>, 128, 16, 128);
  ADD_SUB_MODULE(down_block6, std::make_shared<DownBlock>, 128, 16, 128);
  ADD_SUB_MODULE(down_block7, std::make_shared<DownBlock>, 128, 16, 128);
  ADD_SUB_MODULE(down_block8, std::make_shared<DownMaxPool2dBlock>, 128, 16, 128);
  ADD_SUB_MODULE(down_block9, std::make_shared<DownBlock>, 128, 8, 128);

  ADD_SUB_MODULE(resnet_block0, std::make_shared<ResNetBlockV1>, 128, 8);
  ADD_SUB_MODULE(resnet_block1, std::make_shared<ResNetBlockV2>, 128, 8);
  ADD_SUB_MODULE(resnet_block2, std::make_shared<ResNetBlockV1>, 128, 8);
  ADD_SUB_MODULE(resnet_block3, std::make_shared<DownBlock>, 128, 8, 128);
  ADD_SUB_MODULE(resnet_block4, std::make_shared<ResNetBlockV1>, 128, 8);
  ADD_SUB_MODULE(resnet_block5, std::make_shared<DownBlock>, 128, 8, 128);

  ADD_SUB_MODULE(resnet_block6, std::make_shared<ResNetBlockV1>, 128, 8);
  ADD_SUB_MODULE(resnet_block7, std::make_shared<ResNetBlockV2>, 128, 8);
  ADD_SUB_MODULE(resnet_block8, std::make_shared<ResNetBlockV1>, 128, 8);
  ADD_SUB_MODULE(resnet_block9, std::make_shared<DownBlock>, 128, 8, 128);
  ADD_SUB_MODULE(resnet_block10, std::make_shared<ResNetBlockV1>, 128, 8);
  ADD_SUB_MODULE(resnet_block11, std::make_shared<ResNetBlockV1>, 128, 4);

  ADD_SUB_MODULE(up_block0, std::make_shared<UpMaxPool2dBlock>, 128, 8, 128);
  ADD_SUB_MODULE(up_block1, std::make_shared<UpBlock>, 256, 8, 128);
  ADD_SUB_MODULE(up_block2, std::make_shared<UpMaxPool2dBlock>, 128, 8, 64);
  ADD_SUB_MODULE(up_block3, std::make_shared<UpBlock>, 128, 4, 64);
  ADD_SUB_MODULE(up_block4, std::make_shared<UpMaxPool2dBlock>, 64, 4, 32);
  ADD_SUB_MODULE(up_block5, std::make_shared<ResNetBlockV1>, 32, 4);

  ADD_SUB_MODULE(output_block, sequential, {
    conv_tranpose2d(32, 8, 2, 2, 0, 0, true),
    prelu(true, 8),
    conv_tranpose2d(8, out_channel, 2, 2, 0, 0, true),
  });
}

Tensor HairSeg::forward(Tensor x){
  x = (*input_block)(x);

  auto down_block0_out = down_block0->compute(x);
  auto down_out0 = down_block0_out[0];
  auto indices0 = down_block0_out[1];


  std::cout << down_out0 << '\n';
  std::cout << down_out0.sum() << '\n';


  auto down_out1 = (*down_block1)(down_out0);
  auto down_out2 = (*down_block2)(down_out1);

  auto down_block3_out = down_block3->compute(down_out2);
  auto down_out3 = down_block3_out[0];
  auto indices3 = down_block3_out[1];

  auto down_out = (*down_block4)(down_out3);
  down_out = (*down_block5)(down_out);
  down_out = (*down_block6)(down_out);
  down_out = (*down_block7)(down_out);

  auto down_block8_out = down_block8->compute(down_out);
  auto down_out8 = down_block8_out[0];
  auto indices8 = down_block8_out[1];

  auto down_out9 = (*down_block9)(down_out8);

  auto resnet_out = (*resnet_block0)(down_out9);
  resnet_out = (*resnet_block1)(resnet_out);
  resnet_out = (*resnet_block2)(resnet_out);
  resnet_out = (*resnet_block3)(resnet_out);
  resnet_out = (*resnet_block4)(resnet_out);
  resnet_out = (*resnet_block5)(resnet_out);
  resnet_out = (*resnet_block6)(resnet_out);
  resnet_out = (*resnet_block7)(resnet_out);
  resnet_out = (*resnet_block8)(resnet_out);
  resnet_out = (*resnet_block9)(resnet_out);
  resnet_out = (*resnet_block10)(resnet_out);
  resnet_out = (*resnet_block11)(resnet_out);

  auto up_0 = (*up_block0)({ resnet_out, indices8});
  up_0 = up_0.cat(down_out3, 1);

  auto up_1 = (*up_block1)(up_0);

  auto up_2 = (*up_block2)({up_1, indices3});
  up_2 = up_2.cat(down_out0, 1);

  auto up_3 = (*up_block3)(up_2);
  auto up_4 = (*up_block4)({up_3, indices0});
  auto up_5 = (*up_block5)(up_4);

  return (*output_block)(up_5);
}

}