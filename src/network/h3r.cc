#include "module/relu.h"
#include "module/relu6.h"
#include "network/h3r.h"

namespace dlfm::nn::h3r {


long long get_cur_microseconds() {
  auto time_now = std::chrono::system_clock::now();
  auto duration_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(time_now.time_since_epoch());
  return duration_in_ms.count();
}


Block::Block(int64_t in_channels, int64_t out_channels, int64_t expansion, int64_t stride) {
  if (1 == expansion) {
    ADD_SUB_MODULE(conv, sequential, {
      conv2d(in_channels, in_channels, 3, stride, 1, 1, in_channels, false),
      batch_norm2d(in_channels),
      relu6(true),
      conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, false),
      batch_norm2d(out_channels),
    });
  } else {
    int64_t channels = expansion * in_channels;

    ADD_SUB_MODULE(conv, sequential, {
      conv2d(in_channels, channels, 1, 1, 0, 1, 1, false),
      batch_norm2d(channels),
      relu6(true),
      conv2d(channels, channels, 3, stride, 1, 1, channels, false),
      batch_norm2d(channels),
      relu6(true),
      conv2d(channels, out_channels, 1, 1, 0, 1, 1, false),
      batch_norm2d(out_channels)
    });
  }

  residual = (stride == 1) && (in_channels == out_channels);
}

Tensor Block::forward(Tensor x) {
  auto out = (*conv)(x);

  if (residual) {
    out += x;
  }

  return out;
}

MobileNetV2::MobileNetV2(const std::vector<std::vector<int64_t>> &config) {
  int64_t in_channels = config[0][1];

  std::vector<Module> features_nodes;

  features_nodes.emplace_back(sequential({
    conv2d(3, in_channels, 3, 2, 1, 1, 1, false),
    batch_norm2d(in_channels),
    relu6(true)
  }));

  for (int64_t c = 1; c < config.size(); ++c) {
    auto expansion = config[c][0];
    auto out_channels = config[c][1];
    auto blocks = config[c][2];
    auto stride = config[c][3];

    for (int64_t i = 0; i < blocks; ++i) {
      features_nodes.emplace_back(std::make_shared<Block>(in_channels, out_channels, expansion, ((0 == i) ? stride : 1)));

      in_channels = out_channels;
    }
  }

  ADD_SUB_MODULE(features, sequential, features_nodes);
}

Tensor MobileNetV2::forward(Tensor x) {
  auto c2 = x;
  for (int64_t i = 0; i < 4; ++i) {
    c2 = (*(features->sub_modules_[i]))(c2);
  }

  auto c3 = c2;
  for (int64_t i = 4; i < 7; ++i) {
    c3 = (*(features->sub_modules_[i]))(c3);
  }

  auto c4 = c3;
  for (int64_t i = 7; i < 14; ++i) {
    c4 = (*(features->sub_modules_[i]))(c4);
  }

  int64_t height = c2.shape()[-2];
  int64_t width = c2.shape()[-1];

  c3 = c3.interpolate2d({ height , width }, "bilinear", false);
  c4 = c4.interpolate2d({ height , width }, "bilinear", false);

  return c2.cat({ c3, c4 }, 1);
}

BinaryHeadBlock::BinaryHeadBlock(int64_t in_channels, int64_t proj_channels, int64_t out_channels) {
  ADD_SUB_MODULE(layers, sequential, {
    conv2d(in_channels, proj_channels, 1, 1, 0, 1, 1, false),
    batch_norm2d(proj_channels),
    relu(true),
    conv2d(proj_channels, out_channels * 2, 1, 1, 0, 1, 1, false)
  });
}

Tensor BinaryHeadBlock::forward(Tensor input) {
  auto &shape = input.shape();

  int64_t N = shape[0];
  int64_t C = shape[1];
  int64_t H = shape[2];
  int64_t W = shape[3];

  return (*layers)(input).view({ N, 2, -1, H, W });
}

BinaryHeatmap2Coordinate::BinaryHeatmap2Coordinate(float s, int64_t t)
  :stride(s), topk(t) {
}

Tensor BinaryHeatmap2Coordinate::forward(Tensor input) {
  auto input_dims = input.shape().dim_vector();

  std::vector<int64_t> offsets(input_dims.size(), 0);
  offsets[1] = 1;

  auto extents = input_dims;
  extents[1] = 1;

  auto heapmap = input.slice(offsets, extents);

  extents.erase(extents.begin() + 1);

  heapmap = heapmap.reshape(extents);

  int64_t N = extents[0];
  int64_t C = extents[1];
  int64_t H = extents[2];
  int64_t W = extents[3];

  auto score_and_index = heapmap.view({N, C, 1, -1}).topk(topk, -1);

  auto score = score_and_index[0];
  auto index = score_and_index[1];

  auto remainder = index.remainder(W, false).cast(ElementType::from<float>());
  auto divide = index.floor_divide(W, true).cast(ElementType::from<float>());

  auto coord = remainder.cat({ divide }, 2);

  coord *= score.softmax(-1);

  auto out = coord.sum({ -1 });
  out *= stride;

  return out;
}

HeatmapHead::HeatmapHead(
  int64_t in_channels,
  int64_t proj_channels,
  int64_t out_channels,
  float stride,
  int64_t topk) {
  ADD_SUB_MODULE(head, std::make_shared<BinaryHeadBlock>, in_channels, proj_channels, out_channels);
  ADD_SUB_MODULE(decoder, std::make_shared<BinaryHeatmap2Coordinate>, stride, topk);
}

Tensor HeatmapHead::forward(Tensor input) {
  auto output = (*head)(input);
  return (*decoder)(output);
}

FacialLandmarkDetector::FacialLandmarkDetector() {
  mean = Tensor::create({ 3 }, ElementType::from<float>());
  std = Tensor::create({ 3 }, ElementType::from<float>());

  float *mean_ptr = mean.data<float>();
  mean_ptr[0] = 0.485;
  mean_ptr[1] = 0.456;
  mean_ptr[2] = 0.406;

  float *std_ptr = std.data<float>();
  std_ptr[0] = 0.229;
  std_ptr[1] = 0.224;
  std_ptr[2] = 0.225;

  std::vector<std::vector<int64_t>> config = {
    {1,  32, 1, 1},
    {1,  16, 1, 1},
    {6,  24, 2, 2},
    {6,  32, 3, 2},
    {6,  64, 4, 2},
    {6,  96, 3, 1},
  };

  ADD_SUB_MODULE(backbone, std::make_shared<MobileNetV2>, config);
  ADD_SUB_MODULE(heatmap_head, std::make_shared<HeatmapHead>, 152, 152, 106, 4.0, 9);
}

Tensor FacialLandmarkDetector::forward(Tensor x) {
  auto t1 = get_cur_microseconds();

  // x shape [128, 128, 3]
  x /= 255.0;
  x -= mean;
  x /= std;

  x = x.transpose({2, 0, 1}).reshape({1, 3, 128, 128});

  auto t2 = get_cur_microseconds();

  auto y = (*backbone)(x);

  auto t3 = get_cur_microseconds();

  auto out = (*heatmap_head)(y);

  auto t4 = get_cur_microseconds();

  // std::cout << "t2-t1" << t2 - t1 << "\n";
  // std::cout << "t3-t2" << t3 - t2 << "\n";
  // std::cout << "t4-t3" << t4 - t3 << "\n";

  return out;
}

}