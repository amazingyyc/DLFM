#ifndef NN_H3R_H
#define NN_H3R_H

#include "module/module.h"
#include "module/sequential.h"
#include "module/conv2d.h"
#include "module/batch_norm2d.h"
#include "module/linear.h"

namespace dlfm::nn::h3r {

class Block : public ModuleImpl {
public:
  Sequential conv;

  bool residual;

public:
  Block(int64_t in_channels, int64_t out_channels, int64_t expansion=1, int64_t stride=1);

  Tensor forward(Tensor) override;
};

class MobileNetV2: public ModuleImpl {
public:
  Sequential features;

public:
  MobileNetV2(const std::vector<std::vector<int64_t>> &config);

  Tensor forward(Tensor) override;
};

class BinaryHeadBlock : public ModuleImpl {
public:
  Sequential layers;

public:
  BinaryHeadBlock(int64_t in_channels, int64_t proj_channels, int64_t out_channels);

  Tensor forward(Tensor) override;
};

class BinaryHeatmap2Coordinate : public ModuleImpl {
public:
  int64_t topk;
  float stride;

public:
  BinaryHeatmap2Coordinate(float stride,int64_t topk);

  Tensor forward(Tensor) override;
};

class HeatmapHead: public ModuleImpl {
public:
  std::shared_ptr<BinaryHeadBlock> head;
  std::shared_ptr<BinaryHeatmap2Coordinate> decoder;

public:
  HeatmapHead(
    int64_t in_channels,
    int64_t proj_channels,
    int64_t out_channels,
    float stride,
    int64_t topk);

  Tensor forward(Tensor) override;
};

class FacialLandmarkDetector: public ModuleImpl {
public:
  Tensor mean;
  Tensor std;

  std::shared_ptr<MobileNetV2> backbone;
  std::shared_ptr<HeatmapHead> heatmap_head;

public:
  FacialLandmarkDetector();

  Tensor forward(Tensor) override;
};


}

#endif
