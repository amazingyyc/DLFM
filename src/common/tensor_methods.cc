#include "common/tensor.h"

#include "math/fill.h"
#include "math/relu.h"
#include "math/sigmoid.h"
#include "math/tanh.h"
#include "math/sqrt.h"
#include "math/square.h"
#include "math/unary_cwise.h"
#include "math/binary_cwise.h"
#include "math/cast.h"
#include "math/transpose.h"
#include "math/pad.h"
#include "math/cat.h"
#include "math/conv_transpose2d.h"
#include "math/max_pooling2d.h"
#include "math/max_unpooling2d.h"
#include "math/avg_pooling2d.h"
#include "math/adaptive_avg_pooling2d.h"
#include "math/adaptive_max_pooling2d.h"
#include "math/upsample2d.h"
#include "math/matmul.h"
#include "math/mean.h"
#include "math/sum.h"
#include "math/slice.h"
#include "math/reverse.h"
#include "math/reflection_pad2d.h"
#include "math/clamp.h"
#include "math/instance_norm2d.h"
#include "math/batch_norm2d.h"
#include "math/conv2d.h"
#include "math/var.h"
#include "math/img_mask.h"
#include "math/softmax.h"
#include "math/norm2d.h"
#include "math/log.h"
#include "math/topk.h"

#ifdef HAS_NNPACK
#include "nnpack.h"
#endif

namespace dlfm {

Tensor Tensor::create(const std::vector<int64_t> &dims, ElementType type) {
  Shape shape(dims);

  auto storage = TensorStorage::create(shape.size() * type.byte_width());

  return Tensor(storage, 0, shape, type);
}

Tensor Tensor::create(const Shape &shape, ElementType type) {
  auto storage = TensorStorage::create(shape.size() * type.byte_width());

  return Tensor(storage, 0, shape, type);
}

//std::shared_ptr<TensorStorage>, , Shape, ElementType
Tensor Tensor::create(std::shared_ptr<TensorStorage> storage, size_t offset, const Shape &shape, ElementType type) {
  return Tensor(storage, offset, shape, type);
}

Tensor Tensor::create_from(void* ptr, const std::vector<int64_t> &dims, ElementType type) {
  Shape shape(dims);

  auto storage = TensorStorage::create_from(ptr, shape.size() * type.byte_width());

  return Tensor(storage, 0, shape, type);
}

Tensor Tensor::create_from(void* ptr, const Shape &shape, ElementType type) {
  auto storage = TensorStorage::create_from(ptr, shape.size() * type.byte_width());

  return Tensor(storage, 0, shape, type);
}

Tensor Tensor::zeros(const std::vector<int64_t> &dims, ElementType type) {
  auto tensor = Tensor::create(dims, type);

  tensor.fill(0);

  return tensor;
}

Tensor Tensor::ones(const std::vector<int64_t> &dims, ElementType type) {
  auto tensor = Tensor::create(dims, type);

  tensor.fill(1);

  return tensor;
}

// file format:
// type[int32], rank[int32], dim0, dim1 .. [int32], data....
Tensor Tensor::initialize_from_file(std::string path) {
  ARGUMENT_CHECK(element_type_.is<float>(), "for now initialize_from_file only support float");

  std::ifstream fstream(path, std::fstream::binary | std::fstream::in);

  ARGUMENT_CHECK(fstream.is_open(), "file:" << path << " open error!");

  // read type
  int type_id;
  fstream.read((char*)&type_id, sizeof(type_id));

  ARGUMENT_CHECK(type_id == (int)DType::Float32 || type_id == (int)DType::Float16, "element type error");

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

  ARGUMENT_CHECK(shape == shape_, "shape error, file:" << path << "tensor shape:" << shape_.to_str() << ", file shape:" << shape.to_str());

  if (type_id == (int)DType::Float32) {
    fstream.read((char*)ptr(), num_bytes());
    fstream.close();
  } else {
    auto half_tensor = Tensor::create(shape, ElementType::from<float16>());

    fstream.read((char*)half_tensor.ptr(), half_tensor.num_bytes());
    fstream.close();

    *this = half_tensor.cast(ElementType::from<float>());
  }

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

Tensor Tensor::floor_divide(float val, bool in_place) {
  auto y = *this;

  if (!in_place) {
    y = like();
  }

  math::floor_divide(*this, y, val);

  return y;
}

Tensor Tensor::remainder(float val, bool in_place) {
  auto y = *this;

  if (!in_place) {
    y = like();
  }

  math::remainder(*this, y, val);

  return y;
}

Tensor Tensor::operator[](int64_t idx) {
  ARGUMENT_CHECK(idx >= 0 && idx < shape_[0], "idx out of range");

  auto dims = this->shape_.dim_vector();
  dims.erase(dims.begin());

  return Tensor(storage_, this->offset_ + idx * shape_.stride(0) * element_type_.byte_width(), Shape(dims), element_type_);
}

Tensor Tensor::reshape(const std::vector<int64_t> &dims) {
  Shape to_shape(dims);

  ARGUMENT_CHECK(shape_.size() == to_shape.size(), "reshape need new shape's size must same as this");

  auto target = *this;
  target.shape_.reset(dims);

  return target;
}

Tensor Tensor::reshape(const std::vector<int64_t> &dims) const {
  Shape to_shape(dims);

  ARGUMENT_CHECK(shape_.size() == to_shape.size(), "reshape need new shape's size must same as this");

  auto target = *this;
  target.shape_.reset(dims);

  return target;
}

Tensor Tensor::reshape(const Shape &to_shape) {
  ARGUMENT_CHECK(shape_.size() == to_shape.size(), "reshape need new shape's size must same as this");

  auto target = *this;
  target.shape_ = to_shape;

  return target;
}

Tensor Tensor::view(std::vector<int64_t> dims) {
  int64_t other_size = 1;
  int64_t zero_axis = -1;

  for (int64_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0 && zero_axis != -1) {
      RUNTIME_ERROR("Tensor view need one axis is -1");
    }

    if (dims[i] < 0) {
      zero_axis = i;
    } else {
      other_size *= dims[i];
    }
  }

  if (zero_axis != -1) {
    dims[zero_axis] = shape_.size() / other_size;
  }

  return this->reshape(dims);
}

Tensor Tensor::unsqueeze(size_t axis) {
  auto dims = this->shape().dim_vector();

  ARGUMENT_CHECK(axis <= dims.size(), "unsqueeze out of range");

  dims.insert(dims.begin() + axis, 1);

  return this->reshape(dims);
}

Tensor Tensor::squeeze(size_t axis) {
  auto dims = this->shape().dim_vector();

  ARGUMENT_CHECK(axis < dims.size(), "squeeze out of range");
  ARGUMENT_CHECK(1 == dims[axis], "squeeze axis dimension must be 1");

  dims.erase(dims.begin() + axis);

  return this->reshape(dims);
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

Tensor Tensor::mean(int64_t axis, bool keep_dims) {
  return this->mean(std::vector<int64_t>({ axis }), keep_dims);
}

Tensor Tensor::mean(std::vector<int64_t> axis, bool keep_dims) {
  auto ndims = this->rank();

  if (axis.empty()) {
    for (int64_t i = 0 ; i < ndims; ++i) {
      axis.emplace_back(i);
    }
  }

  std::unordered_set<int64_t> reduce_axis;

  std::vector<int64_t> target_dims_keep;
  std::vector<int64_t> target_dims;

  for (int i = 0; i < axis.size(); ++i) {
    if (axis[i] < 0) {
      axis[i] += ndims;
    }

    ARGUMENT_CHECK(0 <= axis[i] && axis[i] < ndims, "axis out of range");

    reduce_axis.insert(axis[i]);
  }

  for (int i = 0; i < ndims; ++i) {
    if (reduce_axis.find(i) != reduce_axis.end()) {
      target_dims_keep.emplace_back(1);
    } else {
      target_dims_keep.emplace_back(this->shape_[i]);
      target_dims.emplace_back(this->shape_[i]);
    }
  }

  if (target_dims.empty()) {
    target_dims.emplace_back(1);
  }

  auto target = Tensor::create(target_dims_keep, element_type_);

  // shape same, just copy
  if (this->shape() == target.shape()) {
    math::assign(*this, target);

    if (!keep_dims) {
      return target.reshape(target_dims);
    } else {
      return target;
    }
  }

  math::mean(*this, target);

  if (!keep_dims) {
    return target.reshape(target_dims);
  } else {
    return target;
  }
}

Tensor Tensor::sum(std::vector<int64_t> axis, bool keep_dims) {
  auto ndims = this->rank();

  if (axis.empty()) {
    for (int64_t i = 0 ; i < ndims; ++i) {
      axis.emplace_back(i);
    }
  }

  std::unordered_set<int64_t> reduce_axis;

  std::vector<int64_t> target_dims_keep;
  std::vector<int64_t> target_dims;

  for (int i = 0; i < axis.size(); ++i) {
    if (axis[i] < 0) {
      axis[i] += ndims;
    }

    ARGUMENT_CHECK(0 <= axis[i] && axis[i] < ndims, "axis out of range");

    reduce_axis.insert(axis[i]);
  }

  for (int i = 0; i < ndims; ++i) {
    if (reduce_axis.find(i) != reduce_axis.end()) {
      target_dims_keep.emplace_back(1);
    } else {
      target_dims_keep.emplace_back(this->shape_[i]);
      target_dims.emplace_back(this->shape_[i]);
    }
  }

  if (target_dims.empty()) {
    target_dims.emplace_back(1);
  }

  auto target = Tensor::create(target_dims_keep, element_type_);

  // shape same, just copy
  if (this->shape() == target.shape()) {
    math::assign(*this, target);

    if (!keep_dims) {
      return target.reshape(target_dims);
    } else {
      return target;
    }
  }

  math::sum(*this, target);

  if (!keep_dims) {
    return target.reshape(target_dims);
  } else {
    return target;
  }
}

Tensor Tensor::var(int64_t axis, const Tensor &mean, bool unbiased) {
  auto ndims = this->ndims();

  if (axis < 0) {
    axis += ndims;
  }

  ARGUMENT_CHECK(0 <= axis && axis < ndims, "axis out of range");
  ARGUMENT_CHECK(this->ndims() == mean.ndims(), "var need mean's dimension same");

  for (int64_t i = 0; i < ndims; ++i) {
    if (i == axis) {
      ARGUMENT_CHECK(1 == mean.shape()[i], "mean shape error");
    } else {
      ARGUMENT_CHECK(this->shape()[i] == mean.shape()[i], "mean shape error");
    }
  }

  auto target = Tensor::create(mean.shape(), mean.element_type());

  math::var(*this, mean, axis, unbiased, target);

  return target;
}

Tensor Tensor::var(int64_t axis, bool keep_dims, bool unbiased) {
  auto ndims = this->ndims();

  if (axis < 0) {
    axis += ndims;
  }

  ARGUMENT_CHECK(0 <= axis && axis < ndims, "axis out of range");

  auto mean = this->mean( axis, true);
  auto target = this->var(axis, mean, unbiased);

  if (!keep_dims) {
    std::vector<int64_t> dims;

    for (int64_t i = 0; i < ndims; ++i) {
      if (i != axis) {
        dims.emplace_back(this->shape()[i]);
      }
    }

    return target.reshape(dims);
  } else {
    return target;
  }
}

Tensor Tensor::std(int64_t axis, const Tensor &mean, bool unbiased) {
  return this->var(axis, mean, unbiased).sqrt(true);
}

Tensor Tensor::std(int64_t axis, bool keep_dims, bool unbiased) {
  return this->var(axis, keep_dims, unbiased).sqrt(true);
}

Tensor Tensor::softmax(int64_t axis) {
  auto ndims = this->ndims();

  if (axis < 0) {
    axis += this->ndims();
  }

  ARGUMENT_CHECK(0 <= axis && axis < ndims, "axis out of range");

  auto target = this->like();

  math::softmax(*this, target, axis);

  return target;
}

Tensor Tensor::clamp(float min, float max, bool in_place) {
  if (in_place) {
    math::clamp(*this, *this, min, max);

    return *this;
  } else {
    auto target = this->like();

    math::clamp(*this, target, min, max);

    return target;
  }
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

Tensor Tensor::relu6(bool in_place) {
  if (in_place) {
    math::relu6(*this, *this);

    return *this;
  } else {
    auto target = this->like();

    math::relu6(*this, target);

    return target;
  }
}

Tensor Tensor::prelu(const Tensor &w, bool in_place) {
  if (in_place) {
    math::prelu(*this, w, *this);

    return *this;
  } else {
    auto target = this->like();

    math::prelu(*this, w, target);

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

Tensor Tensor::sqrt(bool in_place) {
  if (in_place) {
    math::sqrt(*this, *this);

    return *this;
  } else {
    auto target = this->like();

    math::sqrt(*this, target);

    return target;
  }
}

Tensor Tensor::square(bool in_place) {
  if (in_place) {
    math::square(*this, *this);

    return *this;
  } else {
    auto target = this->like();

    math::square(*this, target);

    return target;
  }
}

Tensor Tensor::log(bool in_place) {
  if (in_place) {
    math::log(*this, *this);

    return *this;
  } else {
    auto target = this->like();

    math::log(*this, target);

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

Tensor Tensor::slice(int64_t axis, int64_t offset, int64_t extent) {
  int64_t ndims = shape_.ndims();

  ARGUMENT_CHECK(0 <= axis && axis < ndims, "axis out of range");
  ARGUMENT_CHECK(0 <= offset && offset < shape_[axis] && 0 < extent && offset + extent <= shape_[axis], "offset/extent error.");

  std::vector<int64_t> offsets;
  std::vector<int64_t> extents;

  for (int64_t i = 0; i < ndims; ++i) {
    if (i != axis) {
      offsets.emplace_back(0);
      extents.emplace_back(shape_[i]);
    } else {
      offsets.emplace_back(offset);
      extents.emplace_back(extent);
    }
  }

  auto target = Tensor::create(extents, element_type_);

  math::slice(*this, target, offsets, extents);

  return target;
}

Tensor Tensor::slice(std::vector<int64_t> offsets, std::vector<int64_t> extents) {
  auto ndims = this->shape().ndims();

  ARGUMENT_CHECK(offsets.size() == ndims && extents.size() == ndims, "offsets and extents size not correct");

  for (int64_t i = 0; i < ndims; ++i) {
    ARGUMENT_CHECK(offsets[i] >= 0 && offsets[i] < this->shape()[i], "offsets out of range");
    ARGUMENT_CHECK(offsets[i] + extents[i] <= this->shape()[i], "extents out of range");
  }

  auto target = Tensor::create(extents, element_type_);

  math::slice(*this, target, offsets, extents);

  return target;
}

Tensor Tensor::reverse(std::vector<bool> reversed) {
  ARGUMENT_CHECK(reversed.size() == this->shape().size(), "reversed size error");

  auto target = this->like();

  math::reverse(*this, target, reversed);

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

Tensor Tensor::reflection_pad2d(size_t padding) {
  return this->reflection_pad2d({ padding , padding , padding , padding });
}

Tensor Tensor::reflection_pad2d(std::vector<size_t> paddings) {
  ARGUMENT_CHECK(4 == this->shape().rank(), "reflection_pad2d need 4d tensor");
  ARGUMENT_CHECK(4 == paddings.size(), "paddings need 1/4 element");

  auto target_dims = shape_.dim_vector();

  target_dims[3] += (paddings[0] + paddings[1]);
  target_dims[2] += (paddings[2] + paddings[3]);

  auto target = Tensor::create(target_dims, element_type_);

  math::reflection_pad2d(*this, target, paddings);

  return target;
}

Tensor Tensor::cat(const Tensor &other, int64_t axis) {
  std::vector<Tensor> others({ other });

  return cat(others, axis);
}

Tensor Tensor::cat(const std::vector<Tensor> &others, int64_t axis) {
  int64_t ndims = shape_.ndims();

  if (axis < 0) {
    axis += ndims;
  }

  ARGUMENT_CHECK(0 <= axis && axis < ndims, "axis out of range");

  for (auto &item : others) {
    ARGUMENT_CHECK(element_type_ == item.element_type_, "element type must same");
    ARGUMENT_CHECK(ndims == item.shape_.ndims(), "cat need tensor ndims same");

    for (int64_t i = 0; i < ndims; ++i) {
      if (i != axis) {
        ARGUMENT_CHECK(shape_[i] == item.shape_[i], "shape error");
      }
    }
  }

  std::vector<Tensor> include_this;
  include_this.emplace_back(*this);

  for (auto &item : others) {
    include_this.emplace_back(item);
  }

  std::vector<int64_t> target_dims;

  for (int64_t i = 0; i < ndims; ++i) {
    if (i != axis) {
      target_dims.emplace_back(shape_[i]);
    } else {
      int64_t total = 0;

      for (auto &item : include_this) {
        total += item.shape_[i];
      }

      target_dims.emplace_back(total);
    }
  }

  auto target = Tensor::create(target_dims, element_type_);

  math::cat_v2(include_this, target, axis);

  return target;
}

Tensor Tensor::normalize(Tensor mean, Tensor std, bool in_place) {
  ARGUMENT_CHECK(3 == this->ndims(), "normalize need tensor dimension is 3");
  ARGUMENT_CHECK(1 == mean.ndims() && 1 == std.ndims(), "normalize need mean/std ndims is 1");
  ARGUMENT_CHECK(this->shape()[0] == mean.shape()[0] && this->shape()[0] == std.shape()[0], "normalize need channel same");

  auto channel = this->shape()[0];

  if (in_place) {
    (*this) -= mean.reshape({ channel, 1, 1 });
    (*this) /= std.reshape({ channel, 1, 1 });

    return *this;
  } else {
    auto target = (*this) - mean.reshape({ channel, 1, 1 });
    target /= std.reshape({ channel, 1, 1 });

    return target;
  }
}

// (this * std) + mean
Tensor Tensor::denormalize(Tensor mean, Tensor std, bool in_place) {
  ARGUMENT_CHECK(3 == this->ndims(), "normalize need tensor dimension is 3");
  ARGUMENT_CHECK(1 == mean.ndims() && 1 == std.ndims(), "normalize need mean/std ndims is 1");
  ARGUMENT_CHECK(this->shape()[0] == mean.shape()[0] && this->shape()[0] == std.shape()[0], "normalize need channel same");

  auto channel = this->shape()[0];

  if (in_place) {
    (*this) *= std.reshape({ channel, 1, 1 });
    (*this) += mean.reshape({ channel, 1, 1 });

    return *this;
  } else {
    auto target = (*this) * std.reshape({ channel, 1, 1 });
    target += mean.reshape({ channel, 1, 1 });

    return target;
  }
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

std::vector<Tensor> Tensor::max_pooling2d_with_indices(std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding, bool ceil_mode) {
  ARGUMENT_CHECK(4 == shape_.ndims(), "max_pooling 4d tensor");
  ARGUMENT_CHECK(element_type_.is<float>(), "max pooling need float");

  int64_t batch_size   = shape_[0];
  int64_t channel      = shape_[1];
  int64_t input_height = shape_[2];
  int64_t input_width  = shape_[3];

  int64_t output_height = (input_height + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
  int64_t output_width = (input_width + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;

  if (ceil_mode) {
    output_height = (int64_t)ceil(1.0 * (input_height + 2 * padding[0] - kernel_size[0]) / stride[0] + 1.0);
    output_width  = (int64_t)ceil(1.0 * (input_width  + 2 * padding[1] - kernel_size[1]) / stride[1] + 1.0);
  }

  auto output = Tensor::create({ batch_size, channel, output_height, output_width }, element_type_);
  auto indices = Tensor::create({ batch_size, channel, output_height, output_width }, ElementType::from<int64_t>());

  math::max_pooling2d_with_indices(*this, output, indices, kernel_size, stride, padding);

  return { output , indices };
}


Tensor Tensor::max_unpooling2d(const Tensor &indices, std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding) {
  ARGUMENT_CHECK(this->shape() == indices.shape(), "max_unpooling2d need shape same");
  ARGUMENT_CHECK(4 == shape_.ndims(), "max_pooling 4d tensor");
  ARGUMENT_CHECK(element_type_.is<float>(), "max pooling need float");

  int64_t batch_size = shape_[0];
  int64_t channel = shape_[1];
  int64_t input_height = shape_[2];
  int64_t input_width = shape_[3];

  int64_t output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_size[0];
  int64_t output_width  = (input_width  - 1) * stride[1] - 2 * padding[1] + kernel_size[1];

  auto output = Tensor::zeros({ batch_size, channel, output_height, output_width }, element_type_);

  math::max_unpooling2d(*this, indices, output);

  return output;
}

Tensor Tensor::avg_pooling2d(size_t kernel_size, size_t stride, size_t padding, bool ceil_mode) {
  return this->avg_pooling2d({kernel_size, kernel_size}, {stride, stride}, {padding, padding}, ceil_mode);
}

Tensor Tensor::avg_pooling2d(std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding, bool ceil_mode) {
  ARGUMENT_CHECK(4 == shape_.ndims(), "avg_pooling 4d tensor");
  ARGUMENT_CHECK(element_type_.is<float>(), "avg pooling need float");

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

  math::avg_pooling2d(*this, output, kernel_size, stride, padding);

  return output;
}

Tensor Tensor::adaptive_avg_pooling2d(size_t size) {
  return adaptive_avg_pooling2d({size, size});
}

Tensor Tensor::adaptive_avg_pooling2d(std::vector<size_t> size) {
  ARGUMENT_CHECK(4 == this->shape().ndims(), "adaptive_avg_pooling2d need ndims is 4");

  auto target = Tensor::create({shape_[0], shape_[1], (int64_t)size[0], (int64_t)size[1]}, element_type_);

  math::adaptive_avg_pooling2d(*this, target);

  return target;
}

Tensor Tensor::adaptive_max_pooling2d(size_t size) {
  return adaptive_max_pooling2d({size, size});
}

Tensor Tensor::adaptive_max_pooling2d(std::vector<size_t> size) {
  ARGUMENT_CHECK(4 == this->shape().ndims(), "adaptive_max_pooling2d need ndims is 4");

  auto target = Tensor::create({shape_[0], shape_[1], (int64_t)size[0], (int64_t)size[1]}, element_type_);

  math::adaptive_max_pooling2d(*this, target);

  return target;
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

Tensor Tensor::interpolate2d(std::vector<int64_t> size, std::string mode, bool align_corners) {
  ARGUMENT_CHECK(4 == this->shape_.rank(), "interpolate2d need rank is 4");
  ARGUMENT_CHECK(size[0] > 0 && size[1] > 0, "size need > 0");

  if ("nearest" == mode) {
    ARGUMENT_CHECK(false == align_corners, "nearest mode only support align_corners is false")
  }

  std::vector<int64_t> output_dims = {shape_[0], shape_[1], size[0], size[1]};

  if (size[0] == shape_[2] && size[1] == shape_[3]) {
    return *this;
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

Tensor Tensor::pixel_shuffle(int64_t upscale_factor) {
  ARGUMENT_CHECK(upscale_factor > 0 && 4 == this->shape_.ndims(), "pixel_shuffle need ndims is 4");

  auto b = this->shape()[0];
  auto c = this->shape()[1];
  auto h = this->shape()[2];
  auto w = this->shape()[3];

  int64_t upscale_factor_squared = upscale_factor * upscale_factor;

  ARGUMENT_CHECK(0 == c % upscale_factor_squared, "pixel_shuffle expects input channel to be divisible by square of upscale_factor*upscale_factor");

  int64_t oc = c / upscale_factor_squared;
  int64_t oh = h * upscale_factor;
  int64_t ow = w * upscale_factor;

  return this->reshape({ b * oc, upscale_factor, upscale_factor , h, w }).transpose({0, 3, 1, 4, 2}).reshape({b, oc, oh, ow });
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
Tensor Tensor::conv2d(
  const Tensor &weight,
  const Tensor &bias,
  const std::vector<size_t> &stride,
  const std::vector<size_t> &padding,
  const std::vector<size_t> &dilation,
  size_t groups) {
  ARGUMENT_CHECK(groups >= 1, "groups must >= 1");
  ARGUMENT_CHECK(4 == shape_.ndims(), "conv2d need shape ndims is 4");
  ARGUMENT_CHECK(0 == shape_[1] % groups && 0 == (weight.shape_[0] % groups), "input/output channel should be divided by groups");
  ARGUMENT_CHECK(4 == weight.shape_.ndims(), "weight ndims must be 4");
  ARGUMENT_CHECK(shape_[1] / groups == weight.shape_[1], "weight shape error");
  ARGUMENT_CHECK(1 == bias.shape_.ndims() && bias.shape_[0] == weight.shape_[0], "bias shape error");
  ARGUMENT_CHECK(2 == stride.size() && stride[0] >= 1 && stride[1] >= 1, "stride error");
  ARGUMENT_CHECK(2 == dilation.size() && dilation[0] >= 1 && dilation[1] >= 1, "stride error");

  int64_t batch = shape_[0];
  int64_t in_height  = shape_[2];
  int64_t in_width   = shape_[3];

  // weight [output_channel, input_channel, kernel_height, kernel_width]
  int64_t out_channel = weight.shape_[0];
  int64_t kernel_height  = weight.shape_[2];
  int64_t kernel_width   = weight.shape_[3];

  int64_t out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) / stride[0] + 1;
  int64_t out_width  = (in_width  + 2 * padding[1] - dilation[1] * (kernel_width  - 1) - 1) / stride[1] + 1;

  auto output = Tensor::create({ batch, out_channel, out_height, out_width }, element_type_);

  math::conv2d(*this, weight, bias, output, stride, padding, dilation, groups);

  return output;
}

// transpose conv2d
Tensor Tensor::conv_transpose2d(
  const Tensor &weight,
  const Tensor &bias,
  const std::vector<size_t> &stride,
  const std::vector<size_t> &padding,
  const std::vector<size_t> &out_padding) {
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

  int64_t kernel_height = weight.shape_[2];
  int64_t kernel_width = weight.shape_[3];

  int64_t out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_height + out_padding[0];
  int64_t out_width  = (in_width  - 1) * stride[1] - 2 * padding[1] + kernel_width + out_padding[1];

  auto output = Tensor::create({batch, out_channel, out_height, out_width}, element_type_);

  math::conv_transpose2d(
    *this,
    weight,
    bias,
    output,
    stride,
    padding,
    out_padding);

  return output;
}

Tensor Tensor::instance_norm2d(float eps) {
  ARGUMENT_CHECK(4 == this->shape_.rank(), "instance_norm2d need rank is 4");

  auto output = this->like();

  math::instance_norm2d(*this, eps, output);

  return output;
}

Tensor Tensor::instance_norm2d(const Tensor &scale, const Tensor &shift, float eps) {
  ARGUMENT_CHECK(4 == this->shape_.rank(), "instance_norm2d need rank is 4");
  ARGUMENT_CHECK(1 == scale.rank() && 1 == shift.rank(), "instance_norm2d need scale/shift rank is 1");
  ARGUMENT_CHECK(shape_[1] == scale.shape()[0] && shape_[1] == shift.shape()[0], "shape error");

  auto output = this->like();

  math::instance_norm2d(*this, scale, shift, eps, output);

  return output;
}

Tensor Tensor::batch_norm2d(const Tensor &mean, const Tensor &var, const Tensor &scale, const Tensor &shift, float eps) {
  ARGUMENT_CHECK(4 == this->shape_.rank(), "instance_norm2d need rank is 4");
  ARGUMENT_CHECK(1 == scale.rank() && 1 == shift.rank(), "instance_norm2d need scale/shift rank is 1");
  ARGUMENT_CHECK(shape_[1] == mean.shape()[0] && shape_[1] == var.shape()[0] && shape_[1] == scale.shape()[0] && shape_[1] == shift.shape()[0], "shape error");

  auto output = this->like();

  math::batch_norm2d(*this, mean, var, scale, shift, eps, output);

  return output;
}

Tensor Tensor::img_mask(const Tensor &mask, const Tensor &val) {
  ARGUMENT_CHECK(3 == shape_.ndims(), "shape ndim must be 3");
  ARGUMENT_CHECK(2 == mask.ndims() && 1 == val.ndims(), "mask/val shape error");
  ARGUMENT_CHECK(shape_[-2] == mask.shape()[-2] && shape_[-1] == mask.shape()[-1], "shape error");
  ARGUMENT_CHECK(shape_[0] == val.shape()[0], "shape error");
  ARGUMENT_CHECK(mask.element_type_.is<uint8_t>(), "mask must be uint8_t");
  ARGUMENT_CHECK(element_type_ == val.element_type_, "val element type must be same with input");

  auto target = this->like();

  math::img_mask(*this, mask, val, target);

  return target;
}

Tensor Tensor::norm2d(float eps) {
  ARGUMENT_CHECK(2 == shape_.ndims(), "shape ndim must be 2");

  auto target = this->like();

  math::norm2d(*this, target, eps);

  return target;
}

std::vector<Tensor> Tensor::topk(int64_t k, int64_t axis, bool largest, bool sorted) {
  int64_t ndims = shape_.rank();

  if (axis < 0) {
    axis += ndims;
  }

  ARGUMENT_CHECK(axis + 1 == ndims, "topk only support last dimension");
  ARGUMENT_CHECK(shape_.dim(axis) > k, "k must > axis's dimension");

  auto dims = shape_.dim_vector();
  dims[axis] = k;

  auto y = Tensor::create(dims, element_type_);
  auto indices = Tensor::create(dims, ElementType::from<int64_t>());

  math::topk(*this, y, indices, k, axis, largest, sorted);

  return {y, indices };
}


std::ostream& operator<<(std::ostream& os, const Tensor &t) {
  if (t.element_type().is<float>()) {
    return t.pretty_print<float>(os);
  } else if (t.element_type().is<uint8_t>()) {
    return t.pretty_print<uint8_t>(os);
  } else if (t.element_type().is<int64_t>()) {
    return t.pretty_print<int64_t>(os);
  } else {
    RUNTIME_ERROR("not support type:" << t.element_type().name());
  }

  return os;
}

// operator override
Tensor operator+(float value, const Tensor &x) {
  auto target = Tensor::create(x.shape(), x.element_type());

  math::add(value, x, target);

  return target;
}

Tensor operator-(float value, const Tensor &x) {
  auto target = Tensor::create(x.shape(), x.element_type());

  math::sub(value, x, target);

  return target;
}

Tensor operator*(float value, const Tensor &x) {
  auto target = Tensor::create(x.shape(), x.element_type());

  math::multiply(value, x, target);

  return target;
}

Tensor operator/(float value, const Tensor &x) {
  auto target = Tensor::create(x.shape(), x.element_type());

  math::divide(value, x, target);

  return target;
}

}