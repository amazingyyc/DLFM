#ifndef NN_BLAZE_FACE_H
#define NN_BLAZE_FACE_H

#include "module/module.h"
#include "module/sequential.h"
#include "module/relu.h"
#include "module/conv2d.h"
#include "module/max_pooling2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::blaze_face {

class BlazeBlock: public ModuleImpl {
public:
  int64_t stride;
  int64_t channel_pad;

  MaxPooling2d max_pool;

  Sequential convs;

  Relu act;

  BlazeBlock(int64_t in_channels, int64_t out_channels, int64_t kernel_size=3, int64_t stride=1);

public:
  Tensor forward(Tensor) override;
};

class BlazeFace: public ModuleImpl {
public:
  int64_t num_classes;
  int64_t num_anchors;
  int64_t num_coords;
  float score_clipping_thresh;
  float x_scale;
  float y_scale;
  float h_scale;
  float w_scale;
  float min_score_thresh;
  float min_suppression_threshold;

  // shape []
  Tensor anchor;

  Sequential backbone1;
  Sequential backbone2;

  Conv2d classifier_8;
  Conv2d classifier_16;

  Conv2d regressor_8;
  Conv2d regressor_16;

  BlazeFace();

public:
  // load torch model
  void load_torch_model(
    const std::unordered_map<std::string, Tensor> &tensor_map,
    std::string parent_name_scope = "") override;

  // decode box, ref:https://github.com/hollance/BlazeFace-PyTorch
  Tensor decode_boxes(Tensor &raw_boxes, Tensor &anchors);

  // calcualte insect area of A and B.
  std::vector<float> intersect(const Tensor &A, const std::vector<Tensor> &B);

  std::vector<float> jaccard(const Tensor &A, const std::vector<Tensor> &B);

  // detect face output Tensor vector with box and score, tensor shape [17]
  std::vector<Tensor> detect(Tensor);
};


}

#endif