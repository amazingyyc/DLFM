#include "common/tensor.h"
#include "module/conv2d.h"
#include "module/relu.h"
#include "module/tanh.h"
#include "module/sequential.h"
#include "module/upsample2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "math/instance_norm2d.h"
#include "network/cartoon_face.h"

namespace dlfm::nn::cartoon_face {

ConvBlock::ConvBlock(int64_t in, int64_t out) {
  dim_out = out;

  ADD_SUB_MODULE(ConvBlock1, sequential, {
    instance_norm2d(in, 1e-05, false),
    relu(true),
    reflection_pad2d(1),
    conv2d(in, out / 2, 3, 1, 0, false)
  });

  ADD_SUB_MODULE(ConvBlock2, sequential, {
    instance_norm2d(out / 2, 1e-05, false),
    relu(true),
    reflection_pad2d(1),
    conv2d(out / 2, out / 4, 3, 1, 0, false)
  });

  ADD_SUB_MODULE(ConvBlock3, sequential, {
    instance_norm2d(out / 4, 1e-05, false),
    relu(true),
    reflection_pad2d(1),
    conv2d(out / 4, out / 4, 3, 1, 0, false)
  });

  ADD_SUB_MODULE(ConvBlock4, sequential, {
    instance_norm2d(in, 1e-05, false),
    relu(true),
    conv2d(in, out, 1, 1, 0, false)
  });
}

Tensor ConvBlock::forward(Tensor x) {
  auto residual = x;

  auto x1 = (*ConvBlock1)(x);
  auto x2 = (*ConvBlock2)(x1);
  auto x3 = (*ConvBlock3)(x2);

  auto out = x1.cat(x2, 1).cat(x3, 1);

  if (residual.shape()[1] != dim_out) {
    residual = (*ConvBlock4)(residual);
  }

  return residual + out;
}

HourGlassBlock::HourGlassBlock(int64_t dim_in, int64_t dim_out) {
  ADD_SUB_MODULE(ConvBlock1_1, std::make_shared<ConvBlock>, dim_in, dim_out);
  ADD_SUB_MODULE(ConvBlock1_2, std::make_shared<ConvBlock>, dim_out, dim_out);
  ADD_SUB_MODULE(ConvBlock2_1, std::make_shared<ConvBlock>, dim_out, dim_out);
  ADD_SUB_MODULE(ConvBlock2_2, std::make_shared<ConvBlock>, dim_out, dim_out);
  ADD_SUB_MODULE(ConvBlock3_1, std::make_shared<ConvBlock>, dim_out, dim_out);
  ADD_SUB_MODULE(ConvBlock3_2, std::make_shared<ConvBlock>, dim_out, dim_out);
  ADD_SUB_MODULE(ConvBlock4_1, std::make_shared<ConvBlock>, dim_out, dim_out);
  ADD_SUB_MODULE(ConvBlock4_2, std::make_shared<ConvBlock>, dim_out, dim_out);

  ADD_SUB_MODULE(ConvBlock5, std::make_shared<ConvBlock>, dim_out, dim_out);

  ADD_SUB_MODULE(ConvBlock6, std::make_shared<ConvBlock>, dim_out, dim_out);
  ADD_SUB_MODULE(ConvBlock7, std::make_shared<ConvBlock>, dim_out, dim_out);
  ADD_SUB_MODULE(ConvBlock8, std::make_shared<ConvBlock>, dim_out, dim_out);
  ADD_SUB_MODULE(ConvBlock9, std::make_shared<ConvBlock>, dim_out, dim_out);
}

Tensor HourGlassBlock::forward(Tensor x) {
  auto skip1 = (*ConvBlock1_1)(x);
  auto down1 = skip1.avg_pooling2d(2, 2);
  down1 = (*ConvBlock1_2)(down1);

  auto skip2 = (*ConvBlock1_1)(down1);
  auto down2 = down1.avg_pooling2d(2, 2);
  down2 = (*ConvBlock1_2)(down2);

  auto skip3 = (*ConvBlock1_1)(down2);
  auto down3 = down2.avg_pooling2d(2, 2);
  down3 = (*ConvBlock1_2)(down3);

  auto skip4 = (*ConvBlock1_1)(down3);
  auto down4 = down3.avg_pooling2d(2, 2);
  down4 = (*ConvBlock1_2)(down4);

  auto center = (*ConvBlock5)(down4);

  auto up4 = (*ConvBlock6)(center);
  up4 = up4.upsample2d(2);
  up4 = skip4 + up4;

  auto up3 = (*ConvBlock7)(up4);
  up3 = up3.upsample2d(2);
  up3 = skip3 + up3;

  auto up2 = (*ConvBlock8)(up3);
  up2 = up2.upsample2d(2);
  up2 = skip2 + up2;

  auto up1 = (*ConvBlock9)(up2);
  up1 = up1.upsample2d(2);
  up1 = skip1 + up1;

  return up1;
}

ResnetBlock::ResnetBlock(int64_t dim, bool use_bias) {
  ADD_SUB_MODULE(conv_block, sequential, {
    reflection_pad2d(1),
    conv2d(dim, dim, 3, 1, 0, use_bias),
    instance_norm2d(dim, 1e-05, false),
    relu(true),
    reflection_pad2d(1),
    conv2d(dim, dim, 3, 1, 0, use_bias),
    instance_norm2d(dim, 1e-05, false)
  });
}

Tensor ResnetBlock::forward(Tensor x) {
  return x + (*conv_block)(x);
}

HourGlass::HourGlass(int64_t dim_in, int64_t dim_out, bool res) {
  use_res = res;

  ADD_SUB_MODULE(HG, sequential, {
    std::make_shared<HourGlassBlock>(dim_in, dim_out),
    std::make_shared<ConvBlock>(dim_out, dim_out),
    conv2d(dim_out, dim_out, 1, 1, 0, false),
    instance_norm2d(dim_out, 1e-05, false),
    relu(true)
  });

  ADD_SUB_MODULE(Conv1, conv2d, dim_out, 3, 1, 1);

  if (use_res) {
    ADD_SUB_MODULE(Conv2, conv2d, dim_out, dim_out, 1, 1);
    ADD_SUB_MODULE(Conv3, conv2d, 3, dim_out, 1, 1);
  }
}

Tensor HourGlass::forward(Tensor x) {
  auto ll = (*HG)(x);
  auto tmp_out = (*Conv1)(ll);

  if (use_res) {
    ll = (*Conv2)(ll);
    tmp_out = (*Conv3)(tmp_out);

    return x + ll + tmp_out;
  }

  return tmp_out;
}

AdaLIN::AdaLIN(int64_t num_features, float e) {
  eps = e;
  rho = Tensor::create({1, num_features, 1, 1});
}

void AdaLIN::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scope.empty()) {
    name_scope = torch_name_scope_;
  }

  rho.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "rho" + TORCH_MODEL_FILE_SUFFIX);
}

Tensor AdaLIN::forward(std::vector<Tensor> x) {
  auto input = x[0];
  auto gamma = x[1];
  auto beta  = x[2];

  ARGUMENT_CHECK(4 == input.shape().ndims(), "input dimension must be 4");

  auto b = input.shape()[0];
  auto c = input.shape()[1];
  auto h = input.shape()[2];
  auto w = input.shape()[3];

  // in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
  auto in_mean = input.mean({2, 3}, true);
  auto in_var = input.reshape({b, c, h * w}).var(-1, in_mean.reshape({b, c, 1})).reshape({b, c, 1, 1});
  in_var += eps;

  auto out_in = (input - in_mean) / in_var.sqrt(true);

  // ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
  auto ln_mean = input.mean({1, 2, 3}, true);
  auto ln_var = input.reshape({b, c * h * w}).var(-1, ln_mean.reshape({b, 1})).reshape({b, 1, 1, 1});
  ln_var += eps;

  auto out_ln = (input - ln_mean) / ln_var.sqrt(true);

  auto out = out_in * rho + out_ln * (1.0 - rho);

  out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3);

  return out;
}

SoftAdaLIN::SoftAdaLIN(int64_t num_features, float eps) {
  ADD_SUB_MODULE(norm, std::make_shared<AdaLIN>, num_features, eps);

  ADD_SUB_MODULE(c_gamma, sequential, {
    linear(num_features, num_features),
    relu(true),
    linear(num_features, num_features),
  });

  ADD_SUB_MODULE(c_beta, sequential, {
    linear(num_features, num_features),
    relu(true),
    linear(num_features, num_features),
  });

  ADD_SUB_MODULE(s_gamma, linear, num_features, num_features);
  ADD_SUB_MODULE(s_beta, linear, num_features, num_features);

  w_gamma = Tensor::create({1, num_features});
  w_beta = Tensor::create({1, num_features});
}

void SoftAdaLIN::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scope.empty()) {
    name_scope = torch_name_scope_;
  }

  w_gamma.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "w_gamma" + TORCH_MODEL_FILE_SUFFIX);
  w_beta.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "w_beta" + TORCH_MODEL_FILE_SUFFIX);
}

Tensor SoftAdaLIN::forward(std::vector<Tensor> input) {
  auto x = input[0];
  auto content_features = input[1];
  auto style_features = input[2];

  auto content_gamma = (*c_gamma)(content_features);
  auto content_beta = (*c_beta)(content_features);

  auto style_gamma = (*s_gamma)(style_features);
  auto style_beta = (*s_beta)(style_features);

  auto soft_gamma = (1.0 - w_gamma) * style_gamma + w_gamma * content_gamma;
  auto soft_beta = (1.0 - w_beta) * style_beta + w_beta * content_beta;

  return (*norm)({x, soft_gamma, soft_beta});
}

ResnetSoftAdaLINBlock::ResnetSoftAdaLINBlock(int64_t dim, bool use_bias) {
  ADD_SUB_MODULE(pad1, reflection_pad2d, 1);
  ADD_SUB_MODULE(conv1, conv2d, dim, dim, 3, 1, 0, use_bias);
  ADD_SUB_MODULE(norm1, std::make_shared<SoftAdaLIN>, dim);
  ADD_SUB_MODULE(relu1, relu, true);

  ADD_SUB_MODULE(pad2, reflection_pad2d, 1);
  ADD_SUB_MODULE(conv2, conv2d, dim, dim, 3, 1, 0, use_bias);
  ADD_SUB_MODULE(norm2, std::make_shared<SoftAdaLIN>, dim);
}

Tensor ResnetSoftAdaLINBlock::forward(std::vector<Tensor> input) {
  auto x = input[0];
  auto content_features = input[1];
  auto style_features = input[2];

  auto out = (*pad1)(x);
  out = (*conv1)(out);
  out = (*norm1)({out, content_features, style_features});
  out = (*relu1)(out);

  out = (*pad2)(out);
  out = (*conv2)(out);
  out = (*norm2)({out, content_features, style_features});
  
  return out + x;
}

LIN::LIN(int64_t num_features, float e) {
  eps = e;

  rho = Tensor::create({1, num_features, 1, 1});
  gamma = Tensor::create({1, num_features, 1, 1});
  beta = Tensor::create({1, num_features, 1, 1});
}

void LIN::load_torch_model(std::string model_folder, std::string parent_name_scope) {
  std::string name_scope = parent_name_scope + TORCH_NAME_SCOPE_SEP + torch_name_scope_;

  if (parent_name_scope.empty()) {
    name_scope = torch_name_scope_;
  }

  rho.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "rho" + TORCH_MODEL_FILE_SUFFIX);
  gamma.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "gamma" + TORCH_MODEL_FILE_SUFFIX);
  beta.initialize_from_file(model_folder + FILE_SEP + name_scope + TORCH_NAME_SCOPE_SEP + "beta" + TORCH_MODEL_FILE_SUFFIX);
}

Tensor LIN::forward(Tensor input) {
  ARGUMENT_CHECK(4 == input.ndims(), "input ndims must be 4");

  auto b = input.shape()[0];
  auto c = input.shape()[1];
  auto h = input.shape()[2];
  auto w = input.shape()[3];

  auto in_mean = input.mean({2, 3}, true);
  auto in_var = input.reshape({b, c , h * w}).var(-1, in_mean.reshape({b, c, 1})).reshape({b, c, 1, 1});
  in_var += eps;

  auto out_in = (input - in_mean) / (in_var).sqrt(true);

  auto ln_mean = input.mean({1, 2, 3}, true);
  auto ln_var = input.reshape({b, c * h * w}).var(-1, ln_mean.reshape({b, 1})).reshape({b, 1, 1, 1});
  ln_var += eps;

  auto out_ln = (input - ln_mean) / ln_var.sqrt(true);

  auto out = rho * out_in + (1.0 - rho) * out_ln;
  out = out * gamma + beta;

  return out;
}

CartoonFace::CartoonFace(int64_t ngf, int64_t img_size, bool l) {
  light = l;

  ADD_SUB_MODULE(ConvBlock1, sequential, {
    reflection_pad2d(3),
    conv2d(3, ngf, 7, 1, 0, false),
    instance_norm2d(ngf, 1e-05, false),
    std::make_shared<ReluImpl>(true)
  });

  ADD_SUB_MODULE(HourGlass1, std::make_shared<HourGlass>, ngf, ngf);
  ADD_SUB_MODULE(HourGlass2, std::make_shared<HourGlass>, ngf, ngf);

  ADD_SUB_MODULE(DownBlock1, sequential, {
    reflection_pad2d(1),
    conv2d(ngf, ngf * 2, 3, 2, 0, false),
    instance_norm2d(ngf * 2, 1e-05, false),
    std::make_shared<ReluImpl>(true)
  });

  ADD_SUB_MODULE(DownBlock2, sequential, {
    reflection_pad2d(1),
    conv2d(ngf * 2, ngf * 4, 3, 2, 0, false),
    instance_norm2d(ngf * 4, 1e-05, false),
    std::make_shared<ReluImpl>(true)
  });

  ADD_SUB_MODULE(EncodeBlock1, std::make_shared<ResnetBlock>, ngf * 4);
  ADD_SUB_MODULE(EncodeBlock2, std::make_shared<ResnetBlock>, ngf * 4);
  ADD_SUB_MODULE(EncodeBlock3, std::make_shared<ResnetBlock>, ngf * 4);
  ADD_SUB_MODULE(EncodeBlock4, std::make_shared<ResnetBlock>, ngf * 4);

  ADD_SUB_MODULE(gap_fc, linear, ngf * 4, 1);
  ADD_SUB_MODULE(gmp_fc, linear, ngf * 4, 1);
  ADD_SUB_MODULE(conv1x1, conv2d, ngf * 8, ngf * 4, 1, 1);
  ADD_SUB_MODULE(relu, std::make_shared<ReluImpl>, true);

  if (light) {
    ADD_SUB_MODULE(FC, sequential, {
      linear(ngf * 4, ngf * 4),
      std::make_shared<ReluImpl>(true),
      linear(ngf * 4, ngf * 4),
      std::make_shared<ReluImpl>(true)
    });
  } else {
    ADD_SUB_MODULE(FC, sequential, {
      linear(img_size / 4 * img_size / 4 * ngf * 4, ngf * 4),
      std::make_shared<ReluImpl>(true),
      linear(ngf * 4, ngf * 4),
      std::make_shared<ReluImpl>(true)
    });
  }

  ADD_SUB_MODULE(DecodeBlock1, std::make_shared<ResnetSoftAdaLINBlock>, ngf * 4);
  ADD_SUB_MODULE(DecodeBlock2, std::make_shared<ResnetSoftAdaLINBlock>, ngf * 4);
  ADD_SUB_MODULE(DecodeBlock3, std::make_shared<ResnetSoftAdaLINBlock>, ngf * 4);
  ADD_SUB_MODULE(DecodeBlock4, std::make_shared<ResnetSoftAdaLINBlock>, ngf * 4);

  ADD_SUB_MODULE(UpBlock1, sequential, {
    upsample2d(2),
    reflection_pad2d(1),
    conv2d(ngf * 4, ngf * 2, 3, 1, 0, false),
    std::make_shared<LIN>(ngf * 2),
    std::make_shared<ReluImpl>(true)
  });

  ADD_SUB_MODULE(UpBlock2, sequential, {
    upsample2d(2),
    reflection_pad2d(1),
    conv2d(ngf * 2, ngf, 3, 1, 0, false),
    std::make_shared<LIN>(ngf),
    std::make_shared<ReluImpl>(true)
  });

  ADD_SUB_MODULE(HourGlass3, std::make_shared<HourGlass>, ngf, ngf);
  ADD_SUB_MODULE(HourGlass4, std::make_shared<HourGlass>, ngf, ngf, false);

  ADD_SUB_MODULE(ConvBlock2, sequential, {
    reflection_pad2d(3),
    conv2d(3, 3, 7, 1, 0, false),
    tanh(true)
  });
}

Tensor CartoonFace::forward(Tensor x) {
  x = (*ConvBlock1)(x);
  x = (*HourGlass1)(x);
  x = (*HourGlass2)(x);

  x = (*DownBlock1)(x);
  x = (*DownBlock2)(x);

  x = (*EncodeBlock1)(x);
  auto content_features1 = x.adaptive_avg_pooling2d(1).view({x.shape()[0], -1});

  x = (*EncodeBlock2)(x);
  auto content_features2 = x.adaptive_avg_pooling2d(1).view({x.shape()[0], -1});

  x = (*EncodeBlock3)(x);
  auto content_features3 = x.adaptive_avg_pooling2d(1).view({x.shape()[0], -1});
  
  x = (*EncodeBlock4)(x);
  auto content_features4 = x.adaptive_avg_pooling2d(1).view({x.shape()[0], -1});

  auto gap = x.adaptive_avg_pooling2d(1);
  // auto gap_logit = (*gap_fc)(gap.view({x.shape()[0], -1}));
  auto gap_weight = gap_fc->weight;
  gap = x * gap_weight.unsqueeze(2).unsqueeze(3);

  auto gmp = x.adaptive_max_pooling2d(1);
  // auto gmp_logit = (*gmp_fc)(gmp.view({x.shape()[0], -1}));
  auto gmp_weight = gmp_fc->weight;
  gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3);

  // auto cam_logit = gap_logit.cat(gmp_logit, 1);
  x = gap.cat(gmp, 1);
  x = (*relu)((*conv1x1)(x));

  // auto heatmap = x.sum(1, true);
  Tensor style_features;

  if (light) {
    auto x_ = x.adaptive_avg_pooling2d(1);
    style_features = (*FC)(x_.view({x_.shape()[0], -1}));
  } else {
    style_features = (*FC)(x.view({x.shape()[0], -1}));
  }

  x = (*DecodeBlock1)({x, content_features4, style_features});
  x = (*DecodeBlock2)({x, content_features3, style_features});
  x = (*DecodeBlock3)({x, content_features2, style_features});
  x = (*DecodeBlock4)({x, content_features1, style_features});

  x = (*UpBlock1)(x);
  x = (*UpBlock2)(x);

  x = (*HourGlass3)(x);
  x = (*HourGlass4)(x);
  auto out = (*ConvBlock2)(x);

  return out;
}

}