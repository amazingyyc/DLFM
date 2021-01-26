#include "network/segnet.h"

namespace dlfm::nn::segnet {

MakeDense::MakeDense(int64_t n_channels, int64_t growth_rate) {
  ADD_SUB_MODULE(conv, conv2d, n_channels, growth_rate, 3, 1, 1, 1, 1, false);
  ADD_SUB_MODULE(bn, batch_norm2d, growth_rate);
  ADD_SUB_MODULE(act, relu, true);
}

Tensor MakeDense::forward(Tensor x) {
  auto out = (*conv)(x);
  out = (*bn)(out);
  out = (*act)(out);

  return x.cat(out, 1);
}

DenseBlock::DenseBlock(int64_t n_channels, int64_t n_denselayer, int64_t growth_rate, bool reset_channel) {
  std::vector<Module> modules;

  for (int64_t i = 0; i < n_denselayer; ++i) {
    modules.emplace_back(std::make_shared<MakeDense>(n_channels, growth_rate));

    n_channels += growth_rate;
  }

  ADD_SUB_MODULE(dense_layers, sequential, modules);
}

Tensor DenseBlock::forward(Tensor x) {
  return (*dense_layers)(x);
}

ResidualDenseBlock::ResidualDenseBlock(int64_t n_in, int64_t s, bool a) {
  int64_t n = n_in / s;

  add = a;

  ADD_SUB_MODULE(conv, conv2d, n_in, n, 1, 1, 0, 1, 1, false);
  ADD_SUB_MODULE(dense_block, std::make_shared<DenseBlock>, n, (s - 1), n);
  ADD_SUB_MODULE(bn, batch_norm2d, n_in);
  ADD_SUB_MODULE(act, prelu, true, n_in);
}

Tensor ResidualDenseBlock::forward(Tensor x) {
  auto combine = (*conv)(x);
  combine = (*dense_block)(combine);

  if (add) {
    combine += x;
  }

  return (*act)((*bn)(combine));
}

InputProjection::InputProjection(int64_t s) : sampling_times(s) {
}

Tensor InputProjection::forward(Tensor x) {
  auto y = x;

  for (int64_t i = 0; i < sampling_times; ++i) {
    y = y.avg_pooling2d(3, 2, 1);
  }

  return y;
}

ERDSegNet::ERDSegNet(int64_t classes) {
  ADD_SUB_MODULE(cascade1, std::make_shared<InputProjection>, 1);
  ADD_SUB_MODULE(cascade2, std::make_shared<InputProjection>, 2);
  ADD_SUB_MODULE(cascade3, std::make_shared<InputProjection>, 3);
  ADD_SUB_MODULE(cascade4, std::make_shared<InputProjection>, 4);

  ADD_SUB_MODULE(head_conv, sequential, conv_bn_act(3, 12, 3, 2, 1));
  ADD_SUB_MODULE(stage_0, std::make_shared<ResidualDenseBlock>, 12, 3, true);

  ADD_SUB_MODULE(ba_1, sequential, bn_act(12 + 3));
  ADD_SUB_MODULE(down_1, sequential, conv_bn_act(12 + 3, 24, 3, 2, 1));
  ADD_SUB_MODULE(stage_1, std::make_shared<ResidualDenseBlock>, 24, 3, true);

  ADD_SUB_MODULE(ba_2, sequential, bn_act(48 + 3));
  ADD_SUB_MODULE(down_2, sequential, conv_bn_act(48 + 3, 48, 3, 2, 1));
  ADD_SUB_MODULE(stage_2, std::make_shared<ResidualDenseBlock>, 48, 3, true);

  ADD_SUB_MODULE(ba_3, sequential, bn_act(96 + 3));
  ADD_SUB_MODULE(down_3, sequential, conv_bn_act(96 + 3, 96, 3, 2, 1));
  ADD_SUB_MODULE(stage_3, sequential, {
    std::make_shared<ResidualDenseBlock>(96, 6, true),
    std::make_shared<ResidualDenseBlock>(96, 6, true),
  });

  ADD_SUB_MODULE(ba_4, sequential, bn_act(192 + 3));
  ADD_SUB_MODULE(down_4, sequential, conv_bn_act(192 + 3, 192, 3, 2, 1));
  ADD_SUB_MODULE(stage_4, sequential, {
    std::make_shared<ResidualDenseBlock>(192, 6, true),
    std::make_shared<ResidualDenseBlock>(192, 6, true),
  });

  ADD_SUB_MODULE(classifier, conv2d, 192, classes, 1, 1, 0, 1, 1, false);

  this->act = nn::prelu(true, classes);
  sub_modules_.emplace_back(this->act);
  this->act->torch_name_scope("prelu");

  ADD_SUB_MODULE(stage3_down, sequential, conv_bn_act(96, classes, 3, 1, 1));
  ADD_SUB_MODULE(bn3, batch_norm2d, classes);
  ADD_SUB_MODULE(conv_3, conv2d, classes, classes, 3, 1, 1, 1, 1, false);

  ADD_SUB_MODULE(stage2_down, sequential, conv_bn_act(48, classes, 3, 1, 1));
  ADD_SUB_MODULE(bn2, batch_norm2d, classes);
  ADD_SUB_MODULE(conv_2, conv2d, classes, classes, 3, 1, 1, 1, 1, false);

  ADD_SUB_MODULE(stage1_down, sequential, conv_bn_act(24, classes, 3, 1, 1));
  ADD_SUB_MODULE(bn1, batch_norm2d, classes);
  ADD_SUB_MODULE(conv_1, conv2d, classes, classes, 3, 1, 1, 1, 1, false);

  ADD_SUB_MODULE(stage0_down, sequential, conv_bn_act(12, classes, 3, 1, 1));
  ADD_SUB_MODULE(bn0, batch_norm2d, classes);
  ADD_SUB_MODULE(conv_0, conv2d, classes, classes, 3, 1, 1, 1, 1, false);
}

std::vector<Module> ERDSegNet::conv_bn_act(int64_t inp, int64_t oup, int64_t kernel_size, int64_t stride, int64_t padding) {
  return std::vector<Module>({
    conv2d(inp, oup, kernel_size, stride, padding, 1, 1, false),
    batch_norm2d(oup),
    std::make_shared<PReluImpl>(true, oup)
  });
}

std::vector<Module> ERDSegNet::bn_act(int64_t inp) {
  return std::vector<Module>({
    batch_norm2d(inp),
    std::make_shared<PReluImpl>(true, inp)
  });
}


long long get_cur_microseconds() {
  auto time_now = std::chrono::system_clock::now();
  auto duration_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(time_now.time_since_epoch());
  return duration_in_ms.count();
}


Tensor ERDSegNet::forward(Tensor input) {
  auto input_cascade1 = (*cascade1)(input);
  auto input_cascade2 = (*cascade2)(input);
  auto input_cascade3 = (*cascade3)(input);
  auto input_cascade4 = (*cascade4)(input);

  auto x = (*head_conv)(input);
  // 1 / 2
  auto s0 = (*stage_0)(x);

  // ---------------
  auto s1_0 = (*down_1)((*ba_1)(input_cascade1.cat({s0}, 1)));
  auto s1 = (*stage_1)(s1_0);

  // ---------------
  auto s2_0 = (*down_2)((*ba_2)(input_cascade2.cat({s1_0, s1}, 1)));
  auto s2 = (*stage_2)(s2_0);

  // ---------------
  auto s3_0 = (*down_3)((*ba_3)(input_cascade3.cat({s2_0, s2}, 1)));
  auto s3 = (*stage_3)(s3_0);

  // ---------------
  auto s4_0 = (*down_4)((*ba_4)(input_cascade4.cat({s3_0, s3}, 1)));
  auto s4 = (*stage_4)(s4_0);

  auto heatmap = (*classifier)(s4);

  auto heatmap_3 = heatmap.upsample2d(2, "bilinear");
  auto s3_heatmap = (*act)((*bn3)((*stage3_down)(s3)));
  heatmap_3 += s3_heatmap;
  heatmap_3 = (*conv_3)(heatmap_3);

  auto heatmap_2 = heatmap_3.upsample2d(2, "bilinear");
  auto s2_heatmap = (*act)((*bn2)((*stage2_down)(s2)));
  heatmap_2 += s2_heatmap;
  heatmap_2 = (*conv_2)(heatmap_2);

  auto heatmap_1 = heatmap_2.upsample2d(2, "bilinear");
  auto s1_heatmap = (*act)((*bn1)((*stage1_down)(s1)));
  heatmap_1 += s1_heatmap;
  heatmap_1 = (*conv_1)(heatmap_1);

  auto heatmap_0 = heatmap_1.upsample2d(2, "bilinear");
  auto s0_heatmap = (*act)((*bn0)((*stage0_down)(s0)));
  heatmap_0 += s0_heatmap;
  heatmap_0 = (*conv_0)(heatmap_0);

  return heatmap_0.upsample2d(2, "bilinear");
}

SegMattingNet::SegMattingNet() {
  ADD_SUB_MODULE(seg_extract, std::make_shared<ERDSegNet>, 2);
  ADD_SUB_MODULE(convF1, conv2d, 11, 8, 3, 1, 1, 1, 1, true);
  ADD_SUB_MODULE(bn, batch_norm2d, 8);
  ADD_SUB_MODULE(convF2, conv2d, 8, 3, 3, 1, 1, 1, 1, true);
}

Tensor SegMattingNet::forward(Tensor x) {
  auto seg = (*seg_extract)(x);

  auto seg_softmax = seg.softmax(1);

  auto bg = seg_softmax.slice(1, 0, 1);
  auto fg = seg_softmax.slice(1, 1, 1);

  auto img_sqr = x.square(false);
  auto img_masked = x * fg;

  auto conv_in = x.cat({seg_softmax, img_sqr, img_masked}, 1);

  auto newconvF1 = (*bn)((*convF1)(conv_in)).relu(true);
  newconvF1 = (*convF2)(newconvF1);

  auto a = newconvF1.slice(1, 0, 1);
  auto b = newconvF1.slice(1, 1, 1);
  auto c = newconvF1.slice(1, 2, 1);

  a *= fg;
  b *= bg;

  a += b;
  a += c;

  return a.sigmoid(true);
}

}
