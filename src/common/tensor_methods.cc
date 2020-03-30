#include "common/tensor.h"

#include "math/fill.h"
#include "math/relu.h"
#include "math/sigmoid.h"
#include "math/tanh.h"
#include "math/unary_cwise.h"
#include "math/binary_cwise.h"
#include "math/cast.h"
#include "math/transpose.h"
#include "math/pad.h"
#include "math/cat.h"
#include "math/conv_transpose2d.h"
#include "math/max_pooling2d.h"
#include "math/upsample2d.h"
#include "math/matmul.h"

#ifdef HAS_NNPACK
#include "nnpack.h"
#endif

namespace dlfm {

Tensor Tensor::create(std::vector<int64_t> dims, ElementType type) {
  Shape shape(dims);

  auto storage = TensorStorage::create(shape.size() * type.byte_width());

  return Tensor(storage, 0, shape, type);
}

Tensor Tensor::create(Shape& shape, ElementType type) {
  auto storage = TensorStorage::create(shape.size() * type.byte_width());

  return Tensor(storage, 0, shape, type);
}

Tensor Tensor::create_from(void* ptr, std::vector<int64_t> dims, ElementType type) {
  Shape shape(dims);

  auto storage = TensorStorage::create_from(ptr, shape.size() * type.byte_width());

  return Tensor(storage, 0, shape, type);
}

Tensor Tensor::create_from(void* ptr, Shape& shape, ElementType type) {
  auto storage = TensorStorage::create_from(ptr, shape.size() * type.byte_width());

  return Tensor(storage, 0, shape, type);
}

Tensor Tensor::zeros(std::vector<int64_t> dims, ElementType type) {
  auto tensor = Tensor::create(dims, type);

  tensor.fill(0);

  return tensor;
}

Tensor Tensor::ones(std::vector<int64_t> dims, ElementType type) {
  auto tensor = Tensor::create(dims, type);

  tensor.fill(1);

  return tensor;
}

// file format:
// type[int32], rank[int32], dim0, dim1 .. [int32], data....
Tensor Tensor::initialize_from_file(std::string path) {
  std::ifstream fstream(path, std::fstream::binary | std::fstream::in);

  ARGUMENT_CHECK(fstream.is_open(), "file:" << path << " open error!");

  // read type
  int type_id;
  fstream.read((char*)&type_id, sizeof(type_id));

  ARGUMENT_CHECK(type_id == (int)element_type_.id(), "element type error");

  // read rank
  int rank;
  fstream.read((char*)&rank, sizeof(rank));

  std::vector<int64_t> dims(rank);

  for (int i = 0; i < rank; ++i) {
    int cur_dim;
    fstream.read((char*)&cur_dim, sizeof(cur_dim));

    dims[i] = cur_dim;
  }

  Shape shape(dims);

  ARGUMENT_CHECK(shape == shape_, "shape error");

  fstream.read((char*)ptr(), num_bytes());
  fstream.close();

  return *this;
}

// operator override
Tensor Tensor::operator+=(const Tensor &other) {
  ARGUMENT_CHECK(this->element_type() == other.element_type(), "element type must same");
  ARGUMENT_CHECK(other.rank() <= this->rank(), "rank error");

  auto e_shape = other.shape().enlarge(this->rank());

  for (int64_t i = 0; i < this->rank(); ++i) {
    ARGUMENT_CHECK(1 == e_shape[i] || e_shape[i] == this->shape()[i], "shape error");
  }

  math::add(*this, other, *this);

  return *this;
}

Tensor Tensor::operator-=(const Tensor &other) {
  ARGUMENT_CHECK(this->element_type() == other.element_type(), "element type must same");
  ARGUMENT_CHECK(other.rank() <= this->rank(), "rank error");

  auto e_shape = other.shape().enlarge(this->rank());

  for (int64_t i = 0; i < this->rank(); ++i) {
    ARGUMENT_CHECK(1 == e_shape[i] || e_shape[i] == this->shape()[i], "shape error");
  }

  math::sub(*this, other, *this);

  return *this;
}

Tensor Tensor::operator*=(const Tensor &other) {
  ARGUMENT_CHECK(this->element_type() == other.element_type(), "element type must same");
  ARGUMENT_CHECK(other.rank() <= this->rank(), "rank error");

  auto e_shape = other.shape().enlarge(this->rank());

  for (int64_t i = 0; i < this->rank(); ++i) {
    ARGUMENT_CHECK(1 == e_shape[i] || e_shape[i] == this->shape()[i], "shape error");
  }

  math::multiply(*this, other, *this);

  return *this;
}

Tensor Tensor::operator/=(const Tensor &other) {
  ARGUMENT_CHECK(this->element_type() == other.element_type(), "element type must same");
  ARGUMENT_CHECK(other.rank() <= this->rank(), "rank error");

  auto e_shape = other.shape().enlarge(this->rank());

  for (int64_t i = 0; i < this->rank(); ++i) {
    ARGUMENT_CHECK(1 == e_shape[i] || e_shape[i] == this->shape()[i], "shape error");
  }

  math::divide(*this, other, *this);

  return *this;
}

Tensor Tensor::operator+(const Tensor &other) {
  ARGUMENT_CHECK(this->element_type() == other.element_type(), "element type must same");

  int64_t o_rank = std::max<int64_t>(this->rank(), other.rank());

  auto x_shape = this->shape().enlarge(o_rank);
  auto y_shape = other.shape().enlarge(o_rank);

  std::vector<int64_t> z_dims;

  for (int64_t i = 0; i < o_rank; ++i) {
    ARGUMENT_CHECK(x_shape[i] == y_shape[i] || (1 == x_shape[i] || 1 == y_shape[i]), "shape error");

    z_dims.emplace_back(std::max<int64_t>(x_shape[i], y_shape[i]));
  }

  auto z = Tensor::create(z_dims, element_type_);

  math::add(*this, other, z);

  return z;
}

Tensor Tensor::operator-(const Tensor &other) {
  ARGUMENT_CHECK(this->element_type() == other.element_type(), "element type must same");

  int64_t o_rank = std::max<int64_t>(this->rank(), other.rank());

  auto x_shape = this->shape().enlarge(o_rank);
  auto y_shape = other.shape().enlarge(o_rank);

  std::vector<int64_t> z_dims;

  for (int64_t i = 0; i < o_rank; ++i) {
    ARGUMENT_CHECK(x_shape[i] == y_shape[i] || (1 == x_shape[i] || 1 == y_shape[i]), "shape error");

    z_dims.emplace_back(std::max<int64_t>(x_shape[i], y_shape[i]));
  }

  auto z = Tensor::create(z_dims, element_type_);

  math::sub(*this, other, z);

  return z;
}

Tensor Tensor::operator*(const Tensor &other) {
  ARGUMENT_CHECK(this->element_type() == other.element_type(), "element type must same");

  int64_t o_rank = std::max<int64_t>(this->rank(), other.rank());

  auto x_shape = this->shape().enlarge(o_rank);
  auto y_shape = other.shape().enlarge(o_rank);

  std::vector<int64_t> z_dims;

  for (int64_t i = 0; i < o_rank; ++i) {
    ARGUMENT_CHECK(x_shape[i] == y_shape[i] || (1 == x_shape[i] || 1 == y_shape[i]), "shape error");

    z_dims.emplace_back(std::max<int64_t>(x_shape[i], y_shape[i]));
  }

  auto z = Tensor::create(z_dims, element_type_);

  math::multiply(*this, other, z);

  return z;
}

Tensor Tensor::operator/(const Tensor &other) {
  ARGUMENT_CHECK(this->element_type() == other.element_type(), "element type must same");

  int64_t o_rank = std::max<int64_t>(this->rank(), other.rank());

  auto x_shape = this->shape().enlarge(o_rank);
  auto y_shape = other.shape().enlarge(o_rank);

  std::vector<int64_t> z_dims;

  for (int64_t i = 0; i < o_rank; ++i) {
    ARGUMENT_CHECK(x_shape[i] == y_shape[i] || (1 == x_shape[i] || 1 == y_shape[i]), "shape error");

    z_dims.emplace_back(std::max<int64_t>(x_shape[i], y_shape[i]));
  }

  auto z = Tensor::create(z_dims, element_type_);

  math::divide(*this, other, z);

  return z;
}

// operator override
Tensor Tensor::operator+=(float value) {
  math::add(*this, value, *this);

  return *this;
}

Tensor Tensor::operator-=(float value) {
  math::sub(*this, value, *this);

  return *this;
}

Tensor Tensor::operator*=(float value) {
  math::multiply(*this, value, *this);

  return *this;
}

Tensor Tensor::operator/=(float value) {
  math::divide(*this, value, *this);

  return *this;
}

Tensor Tensor::operator+(float value) {
  auto target = this->like();

  math::add(*this, value, target);

  return target;
}

Tensor Tensor::operator-(float value) {
  auto target = this->like();

  math::sub(*this, value, target);

  return target;
}

Tensor Tensor::operator*(float value) {
  auto target = this->like();

  math::multiply(*this, value, target);

  return target;
}

Tensor Tensor::operator/(float value) {
  auto target = this->like();

  math::divide(*this, value, target);

  return target;
}

Tensor Tensor::reshape(std::vector<int64_t> dims) {
  Shape to_shape(dims);

  ARGUMENT_CHECK(shape_.size() == to_shape.size(), "reshape need new shape's size must same as this");

  auto target = *this;
  target.shape_.reset(dims);

  return target;
}

Tensor Tensor::reshape(std::vector<int64_t> dims) const {
  Shape to_shape(dims);

  ARGUMENT_CHECK(shape_.size() == to_shape.size(), "reshape need new shape's size must same as this");

  auto target = *this;
  target.shape_.reset(dims);

  return target;
}

Tensor Tensor::reshape(Shape &to_shape) {
  ARGUMENT_CHECK(shape_.size() == to_shape.size(), "reshape need new shape's size must same as this");

  auto target = *this;
  target.shape_ = to_shape;

  return target;
}

// shape and type like this tensor
Tensor Tensor::like() {
  return Tensor::create(shape_, element_type_);
}

Tensor Tensor::clone() {
  auto target = this->like();

  math::assign(*this, target);

  return target;
}


// set all element of this tensor to be value.
Tensor Tensor::fill(float value) {
  math::fill(*this, value);

  return *this;
}

Tensor Tensor::relu(bool in_place) {
  if (in_place) {
    math::relu(*this, *this);

    return *this;
  } else {
    auto target = this->like();

    math::relu(*this, target);

    return target;
  }
}

Tensor Tensor::sigmoid(bool in_place) {
  if (in_place) {
    math::sigmoid(*this, *this);

    return *this;
  } else {
    auto target = this->like();

    math::sigmoid(*this, target);

    return target;
  }
}

Tensor Tensor::tanh(bool in_place) {
  if (in_place) {
    math::tanh(*this, *this);

    return *this;
  } else {
    auto target = this->like();

    math::tanh(*this, target);

    return target;
  }
}


Tensor Tensor::cast(ElementType to_type) {
  if (element_type_ == to_type) {
    return *this;
  }

  auto target = Tensor::create(shape_, to_type);

  math::cast(*this, target);

  return target;
}

Tensor Tensor::transpose(std::vector<size_t> axis) {
  auto ndims = this->ndims();

  ARGUMENT_CHECK(ndims == axis.size(), "the transpose axis is error");

  std::unordered_set<size_t> include;
  std::vector<int64_t> target_dims;

  for (int i = 0; i < ndims; ++i) {
    ARGUMENT_CHECK(0 <= axis[i] && axis[i] < ndims, "transpose axis out of range");
    ARGUMENT_CHECK(include.find(axis[i]) == include.end(), "transpose axis can not include same axis");

    include.insert(axis[i]);

    target_dims.emplace_back(shape_[axis[i]]);
  }

  auto target = Tensor::create(target_dims, element_type_);

  math::transpose(*this, target, axis);

  return target;
}

// pad like pytorh torch.nn.functional.pad(input, pad, mode='constant', value=0)
Tensor Tensor::pad(std::vector<size_t> paddings) {
  ARGUMENT_CHECK(0 == paddings.size() % 2, "paddings must divided by 2");

  auto target_dims = shape_.dim_vector();

  while (paddings.size() < 2 * target_dims.size()) {
    paddings.emplace_back(0);
  }

  auto ndims = shape_.ndims();

  for (int i = 0; i < ndims; ++i) {
    target_dims[ndims - i - 1] += (paddings[2 * i] + paddings[2 * i + 1]);
  }

  auto target = Tensor::create(target_dims, element_type_);

  math::pad(*this, target, paddings);

  return target;
}

Tensor Tensor::cat(const Tensor &other, int64_t axis) {
  ARGUMENT_CHECK(shape_.ndims() == other.ndims(), "cat need 2 tensor ndims same");
  ARGUMENT_CHECK(0 <= axis && axis < other.ndims(), "axis out of range");
  ARGUMENT_CHECK(element_type_ == other.element_type_, "element type must same");

  std::vector<int64_t> target_dims;

  for (int64_t i = 0; i< shape_.ndims(); ++i) {
    if (i == axis) {
      target_dims.emplace_back(shape_[i] + other.shape_[i]);
    } else {
      ARGUMENT_CHECK(shape_[i] == other.shape_[i], "shape error");

      target_dims.emplace_back(shape_[i]);
    }
  }

  auto target = Tensor::create(target_dims, element_type_);

  math::cat(*this, other, target, axis);

  return target;
}

Tensor Tensor::max_pooling2d(std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding, bool ceil_mode) {
  ARGUMENT_CHECK(4 == shape_.ndims(), "max_pooling 4d tensor");
  ARGUMENT_CHECK(element_type_.is<float>(), "max pooling need float");

  int64_t batch_size   = shape_[0];
  int64_t channel      = shape_[1];
  int64_t input_height = shape_[2];
  int64_t input_width  = shape_[3];

  int64_t output_height = (input_height + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
  int64_t output_width  = (input_width  + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;

  if (ceil_mode) {
    output_height = (int64_t)ceil(1.0 * (input_height + 2 * padding[0] - kernel_size[0]) / stride[0] + 1.0);
    output_width  = (int64_t)ceil(1.0 * (input_width  + 2 * padding[1] - kernel_size[1]) / stride[1] + 1.0);
  }

  auto output = Tensor::create({ batch_size, channel, output_height, output_width }, element_type_);

  math::max_pooling2d(*this, output, kernel_size, stride, padding);

  return output;
}

Tensor Tensor::upsample2d(float scale_factor, std::string mode, bool align_corners) {
  ARGUMENT_CHECK(4 == this->shape_.rank(), "upsample2d need rank is 4");

  if ("nearest" == mode) {
    ARGUMENT_CHECK(false == align_corners, "nearest mode only support align_corners is false")
  }

  int64_t output_height = (int64_t)(scale_factor * shape_[2]);
  int64_t output_width  = (int64_t)(scale_factor * shape_[3]);

  std::vector<int64_t> output_dims = {shape_[0], shape_[1], output_height, output_width};

  if (output_height == shape_[2] && output_width == shape_[3]) {
    return this->clone();
  }

  auto output = Tensor::create(output_dims, element_type_);

  if ("nearest" == mode) {
    math::upsample_nearest2d(*this, output);
  } else if ("bilinear" == mode) {
    math::upsample_bilinear2d(*this, output, align_corners);
  } else {
    RUNTIME_ERROR("not support mode");
  }

  return output;
}

Tensor Tensor::matmul(const Tensor &y, bool transpose_a, bool transpose_b) {
  ARGUMENT_CHECK(2 == this->shape_.rank() && 2 == y.shape().rank(), "matmul only support rank is 2");
  ARGUMENT_CHECK(element_type_ == y.element_type(), "matmul need element type same");

  auto xrow = shape_[0];
  auto xcol = shape_[1];

  auto yrow = y.shape()[0];
  auto ycol = y.shape()[1];

  int64_t m, n;

  if (transpose_a && transpose_b) {
    ARGUMENT_CHECK(xrow == ycol, "shape error");

    m = xcol;
    n = yrow;
  } else if (transpose_a) {
    ARGUMENT_CHECK(xrow == yrow, "shape error");

    m = xcol;
    n = ycol;
  } else if (transpose_b) {
    ARGUMENT_CHECK(xcol == ycol, "shape error");

    m = xrow;
    n = yrow;
  } else {
    ARGUMENT_CHECK(xcol == yrow, "shape error");

    m = xrow;
    n = ycol;
  }

  auto z = Tensor::create({m, n}, element_type_);

  math::matmul(*this, y, z, transpose_a, transpose_b);

  return z;
}

// conv2d
Tensor Tensor::conv2d(const Tensor &weight, const Tensor &bias, std::vector<size_t> stride, std::vector<size_t> padding) {
  ARGUMENT_CHECK(3 == shape_.ndims() || (4 == shape_.ndims() && 1 == shape_[0]), "conv2 need shape ndims is 3 or 4 (batch must be 1)");
  ARGUMENT_CHECK(4 == weight.shape_.ndims(), "weight ndims must be 4");
  ARGUMENT_CHECK(shape_[-3] == weight.shape_[1], "input channel must same with weight");
  ARGUMENT_CHECK(1 == bias.shape_.ndims() && bias.shape_[0] == weight.shape_[0], "shape error");

  int64_t input_channel = shape_[-3];
  int64_t input_height  = shape_[-2];
  int64_t input_width   = shape_[-1];

  // weight [output_channel, input_channel, kernel_height, kernel_width]
  int64_t output_channel = weight.shape_[0];
  int64_t kernel_height  = weight.shape_[2];
  int64_t kernel_width   = weight.shape_[3];

  int64_t output_height = (input_height + 2 * padding[0] - kernel_height) / stride[0] + 1;
  int64_t output_width  = (input_width  + 2 * padding[1] - kernel_width)  / stride[1] + 1;

  std::vector<int64_t> output_dims;

  if (4 == shape_.ndims()) {
    output_dims = {1, output_channel, output_height, output_width};
  } else {
    output_dims = {output_channel, output_height, output_width};
  }

  auto output = Tensor::create(output_dims, element_type_);

#ifdef HAS_NNPACK
  auto nnp_status = nnp_convolution_inference(
          nnp_convolution_algorithm_auto,
    nnp_convolution_transform_strategy_block_based,
    input_channel,
    output_channel,
    {.width=(size_t)input_width, .height=(size_t)input_height},
    {.top=padding[0], .right=padding[1], .bottom=padding[0], .left=padding[1]},
    {.width=(size_t)kernel_width, .height=(size_t)kernel_height},
    {.width=stride[1], .height=stride[0]},
    this->data<float>(),
    weight.data<float>(),
    bias.data<float>(),
    output.data<float>(),
    nnpack_threadpool(),
    nullptr);

  if (nnp_status != nnp_status_success) {
    std::cout << "conv2d get error, status:" << nnp_status << "\n";
  }

#else
  RUNTIME_ERROR("please build with nnpack");
#endif

  return output;
}

// transpose conv2d
Tensor Tensor::conv_transpose2d(const Tensor &weight,
                              const Tensor &bias,
                              std::vector<size_t> stride,
                              std::vector<size_t> padding,
                              std::vector<size_t> out_padding) {
  ARGUMENT_CHECK(4 == shape_.ndims(), "shape error");
  ARGUMENT_CHECK(4 == weight.shape_.ndims(), "shape error");
  ARGUMENT_CHECK(1 == bias.shape_.ndims(), "shape error");
  ARGUMENT_CHECK(shape_[1] == weight.shape_[0], "shape error");
  ARGUMENT_CHECK(bias.shape_[0] == weight.shape_[1], "shape error");
  ARGUMENT_CHECK(1 == shape_[0], "conv_transpose2d need batch is 1");

  int64_t batch = shape_[0];
  int64_t in_channel = shape_[1];
  int64_t in_height  = shape_[2];
  int64_t in_width   = shape_[3];

  int64_t out_channel = weight.shape_[1];

  int64_t kernel_size_0 = weight.shape_[2];
  int64_t kernel_size_1 = weight.shape_[3];

  int64_t out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_size_0 + out_padding[0];
  int64_t out_width  = (in_width  - 1) * stride[1] - 2 * padding[1] + kernel_size_1 + out_padding[1];

  auto output = Tensor::create({batch, out_channel, out_height, out_width}, element_type_);

  math::conv_transpose2d(*this,
                         weight,
                         bias,
                         output,
                         {(size_t)kernel_size_0, (size_t)kernel_size_1},
                         stride,
                         padding,
                         out_padding);

  return output;
}

std::ostream& operator<<(std::ostream& os, const Tensor &t) {
  if (t.element_type().is<float>()) {
    return t.pretty_print<float>(os);
  } else if (t.element_type().is<uint8_t>()) {
    return t.pretty_print<uint8_t>(os);
  } else {
    RUNTIME_ERROR("not support type:" << t.element_type().name());
  }

  return os;
}

}