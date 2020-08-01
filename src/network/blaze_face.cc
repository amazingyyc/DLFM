#include "module/conv2d.h"
#include "module/relu.h"
#include "module/tanh.h"
#include "module/sequential.h"
#include "module/upsample2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "math/instance_norm2d.h"
#include "network/blaze_face.h"

namespace dlfm::nn::blaze_face {

BlazeBlock::BlazeBlock(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t s) {
  stride = s;
  channel_pad = out_channels - in_channels;

  int64_t padding = 0;
  if (2 == stride) {
    max_pool = max_pooling2d((size_t)stride, (size_t)stride, 0);
    padding = 0;
  } else {
    padding = (kernel_size - 1) / 2;
  }

  ADD_SUB_MODULE(convs, sequential, {
    conv2d(in_channels, in_channels, kernel_size, stride, padding, in_channels, true),
    conv2d(in_channels, out_channels, 1, 1, 0, 1, true)});

  act = relu(true);
}

Tensor BlazeBlock::forward(Tensor x) {
  Tensor h = x;
  if (2 == stride) {
    h = x.pad({0, 2, 0, 2});
    x = (*max_pool)(x);
  }

  if (channel_pad > 0) {
    x = x.pad({0, 0, 0, 0, 0, (size_t)channel_pad});
  }

  return (*act)((*convs)(h) + x);
}

BlazeFace::BlazeFace() {
  num_classes = 1;
  num_anchors = 896;
  num_coords = 16;

  score_clipping_thresh = 100.0;
  x_scale = 128.0;
  y_scale = 128.0;
  h_scale = 128.0;
  w_scale = 128.0;
  min_score_thresh = 0.90;
  min_suppression_threshold = 0.3;

  anchor = Tensor::create({num_anchors, 4});

  ADD_SUB_MODULE(backbone1, sequential, {
    conv2d(3, 24, 5, 2, 0, 1, true),
    relu(true),
    std::make_shared<BlazeBlock>(24, 24),
    std::make_shared<BlazeBlock>(24, 28),
    std::make_shared<BlazeBlock>(28, 32, 3, 2),
    std::make_shared<BlazeBlock>(32, 36),
    std::make_shared<BlazeBlock>(36, 42),
    std::make_shared<BlazeBlock>(42, 48, 3, 2),
    std::make_shared<BlazeBlock>(48, 56),
    std::make_shared<BlazeBlock>(56, 64),
    std::make_shared<BlazeBlock>(64, 72),
    std::make_shared<BlazeBlock>(72, 80),
    std::make_shared<BlazeBlock>(80, 88),
  });

  ADD_SUB_MODULE(backbone2, sequential, {
    std::make_shared<BlazeBlock>(88, 96, 3, 2),
    std::make_shared<BlazeBlock>(96, 96),
    std::make_shared<BlazeBlock>(96, 96),
    std::make_shared<BlazeBlock>(96, 96),
    std::make_shared<BlazeBlock>(96, 96),
  });

  ADD_SUB_MODULE(classifier_8, conv2d, 88, 2, 1);
  ADD_SUB_MODULE(classifier_16, conv2d, 96, 6, 1);

  ADD_SUB_MODULE(regressor_8, conv2d, 88, 32, 1);
  ADD_SUB_MODULE(regressor_16, conv2d, 96, 96, 1);
}

void BlazeFace::load_torch_model(std::string model_folder, std::string parent_name_scop) {
  ModuleImpl::load_torch_model(model_folder, parent_name_scop);

  // load anchor
  anchor.initialize_from_file(model_folder + FILE_SEP + torch_name_scope_ + ".anchors" + TORCH_MODEL_FILE_SUFFIX);
}

// return shape: [num_anchors, 16]
Tensor BlazeFace::decode_boxes(Tensor &raw_boxes, Tensor &anchors) {
  ARGUMENT_CHECK(raw_boxes.shape() == Shape({num_anchors, 16}), "raw_boxes shape error");

  auto boxes = Tensor::create(raw_boxes.shape());

  float *raw_boxes_ptr = raw_boxes.data<float>();
  float *boxes_ptr = boxes.data<float>();
  float *anchors_ptr = anchors.data<float>();

  for (int64_t i = 0; i < num_anchors; ++i) {
    float x_center = raw_boxes_ptr[i * 16 + 0] / x_scale * anchors_ptr[i * 4 + 2] + anchors_ptr[i * 4 + 0];
    float y_center = raw_boxes_ptr[i * 16 + 1] / y_scale * anchors_ptr[i * 4 + 3] + anchors_ptr[i * 4 + 1];

    float w = raw_boxes_ptr[i * 16 + 2] / w_scale * anchors_ptr[i * 4 + 2];
    float h = raw_boxes_ptr[i * 16 + 3] / h_scale * anchors_ptr[i * 4 + 3];

    boxes_ptr[i * 16 + 0] = y_center - h / 2; // ymin
    boxes_ptr[i * 16 + 1] = x_center - w / 2; // xmin
    boxes_ptr[i * 16 + 2] = y_center + h / 2; // ymax
    boxes_ptr[i * 16 + 3] = x_center + w / 2; // xmax

    for (int64_t j = 0; j < 6; ++j) {
      int64_t offset = 4 + j * 2;

      boxes_ptr[i * 16 + offset + 0] = raw_boxes_ptr[i * 16 + offset + 0] / x_scale * anchors_ptr[i * 4 + 2] + anchors_ptr[i * 4 + 0];
      boxes_ptr[i * 16 + offset + 1] = raw_boxes_ptr[i * 16 + offset + 1] / y_scale * anchors_ptr[i * 4 + 3] + anchors_ptr[i * 4 + 1];
    }
  }

  return boxes;
}

// calcualte insect area of A and B.
// Tensor shape: [17], first 4 is:[ymin, xmin, ymax, xmax]
std::vector<float> BlazeFace::intersect(const Tensor &A, const std::vector<Tensor> &B) {
  std::vector<float> area;
  area.resize(B.size());

  for (size_t j = 0; j < B.size(); ++j) {
    float max_y = std::min<float>(A.data<float>()[2], B[j].data<float>()[2]);
    float max_x = std::min<float>(A.data<float>()[3], B[j].data<float>()[3]);
    float min_y = std::max<float>(A.data<float>()[0], B[j].data<float>()[0]);
    float min_x = std::max<float>(A.data<float>()[1], B[j].data<float>()[1]);

    area[j] = std::max<float>(0, (max_y - min_y) * (max_x - min_x));
  }

  return std::move(area);
}

std::vector<float> BlazeFace::jaccard(const Tensor &A, const std::vector<Tensor> &B) {
  auto intersect_area = intersect(A, B);

  std::vector<float> accard_overlap;
  accard_overlap.resize(B.size());

  float A_area = (A.data<float>()[2] - A.data<float>()[0]) * (A.data<float>()[3] - A.data<float>()[1]);

  std::vector<float> B_area;
  B_area.resize(B.size());

  for (size_t i = 0; i < B.size(); ++i) {
    float max_y = B[i].data<float>()[2];
    float max_x = B[i].data<float>()[3];
    float min_y = B[i].data<float>()[0];
    float min_x = B[i].data<float>()[1];

    B_area[i] = (max_y - min_y) * (max_x - min_x);
  }

  for (size_t j = 0; j < B.size(); ++j) {
    accard_overlap[j] = intersect_area[j] / (A_area + B_area[j] - intersect_area[j]);
  }

  return std::move(accard_overlap);
}

std::vector<Tensor> BlazeFace::detect(Tensor x) {
  // input must be [1, 3, 128, 128]
  ARGUMENT_CHECK(x.shape() == Shape({1, 3, 128, 128}), "BlazeFace x shape error");
  ARGUMENT_CHECK(x.element_type().is<float>(), "x must be float");

  x = x.pad({1, 2, 1, 2});

  int64_t b = x.shape()[0];

  x = (*backbone1)(x); // (b, 88, 16, 16)
  auto h = (*backbone2)(x); // (b, 96, 8, 8)

  auto c1 = (*classifier_8)(x); // (b, 2, 16, 16)
  c1 = c1.transpose({0, 2, 3, 1}); // (b, 16, 16, 2)
  c1 = c1.reshape({b, 512, 1}); // (b, 512, 1)

  auto c2 = (*classifier_16)(h); // (b, 6, 8, 8)
  c2 = c2.transpose({0, 2, 3, 1}); // (b, 8, 8, 6)
  c2 = c2.reshape({b, 384, 1}); // (b, 384, 1)

  auto c = c1.cat(c2, 1); // (b, 896, 1)

  auto r1 = (*regressor_8)(x); // (b, 32, 16, 16)
  r1 = r1.transpose({0, 2, 3, 1}); // (b, 16, 16, 32)
  r1 = r1.reshape({b, 512, 16}); // (b, 512, 16)

  auto r2 = (*regressor_16)(h); // (b, 96, 8, 8)
  r2 = r2.transpose({0, 2, 3, 1}); // (b, 8, 8, 96)
  r2 = r2.reshape({b, 384, 16}); // (b, 384, 16)

  auto r = r1.cat(r2, 1);  // (b, 896, 16)

  // c: score
  // r: coordinate
  // change shape to [896]/[896, 16]
  c = c.reshape({num_anchors, 1});
  r = r.reshape({num_anchors, 16});

  c.clamp(-score_clipping_thresh, score_clipping_thresh, true);

  auto detection_boxes = decode_boxes(r, anchor);
  auto detection_scores = c.sigmoid(true);

  // check qualified box.
  std::vector<Tensor> remaining;

  for (int64_t i = 0; i < num_anchors; ++i) {
    if (detection_scores.data<float>()[i] > min_score_thresh) {
      remaining.emplace_back(detection_boxes[i].cat(detection_scores[i], 0));
    }
  }

  if (remaining.empty()) {
    return {};
  }

  // qualified include tensor: [, 17(box + score)]
  // sort by score
  std::sort(remaining.begin(), remaining.end(), [](const Tensor &t1, const Tensor &t2) -> bool {
    return t1.data<float>()[16] > t2.data<float>()[16];
  });

  std::vector<Tensor> output_detections;

  while (!remaining.empty()) {
    auto ious = jaccard(remaining[0], remaining);

    std::vector<Tensor> overlapping;
    std::vector<Tensor> left;

    for (size_t i = 0; i < remaining.size(); ++i) {
      if (ious[i] > min_suppression_threshold) {
        overlapping.push_back(remaining[i]);
      } else {
        left.push_back(remaining[i]);
      }
    }

    remaining = std::move(left);

    auto weighted_detection = Tensor::zeros({17});

    float total_score = 0;

    for (size_t i = 0; i < overlapping.size(); ++i) {
      total_score += overlapping[i].data<float>()[16];
    }

    for (size_t i = 0; i < overlapping.size(); ++i) {
      for (size_t j = 0; j < 16; ++j) {
        weighted_detection.data<float>()[j] += overlapping[i].data<float>()[j] * overlapping[i].data<float>()[16];
      }
    }

    for (size_t j = 0; j < 16; ++j) {
      weighted_detection.data<float>()[j] /= total_score;
    }

    weighted_detection.data<float>()[16] = total_score / float(overlapping.size());

    output_detections.emplace_back(weighted_detection);
  }

  return output_detections;
}

}