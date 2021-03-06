#include "common/exception.h"
#include "common/shape.h"

namespace dlfm {

Shape::Shape() {}

Shape::Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {
  for (auto &d : dims_) {
    ARGUMENT_CHECK(d > 0, "dimension must > 0");
  }

  update_strides();
}

void Shape::update_strides() {
  for (auto &d : dims_) {
    ARGUMENT_CHECK(d > 0, "dimension must > 0");
  }

  int64_t count = rank();

  strides_.resize(count);

  if (count > 0) {
    strides_[count - 1] = 1;

    for (int64_t i = count - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * dims_[i + 1];
    }
  }
}

Shape &Shape::operator=(const Shape &other) {
  int64_t rank = (int64_t)other.dims_.size();

  dims_.resize(rank);
  strides_.resize(rank);

  for (int64_t i = 0; i < rank; ++i) {
    dims_[i] = other.dims_[i];
    strides_[i] = other.strides_[i];
  }

  return *this;
}

bool Shape::operator==(const Shape &other) const {
  if (dims_.size() != other.dims_.size()) {
    return false;
  }

  for (int64_t i = 0; i < (int64_t)dims_.size(); ++i) {
    if (dims_[i] != other.dims_[i]) {
      return false;
    }
  }

  return true;
}

bool Shape::operator!=(const Shape &other) const {
  return !((*this) == other);
}

int64_t Shape::operator[](int64_t axis) const {
  return dim(axis);
}

void Shape::reset(std::vector<int64_t> dims) {
  for (auto &d : dims) {
    ARGUMENT_CHECK(d > 0, "dimension must > 0");
  }

  dims_ = std::move(dims);

  update_strides();
}

bool Shape::is_scalar() const {
  return 1 == size();
}

int64_t Shape::ndims() const {
  return rank();
}

int64_t Shape::rank() const {
  return (int64_t)dims_.size();
}

int64_t Shape::size() const {
  int64_t size = 1;

  for (auto &d : dims_) {
    size *= d;
  }

  return size;
}

int64_t Shape::dim(int64_t axis) const {
  if (axis < 0) {
    axis += rank();
  }

  ARGUMENT_CHECK(0 <= axis && axis < rank(),
                 "the axis is out of rang: [0, " << rank() << "]");

  return dims_[axis];
}

std::vector<int64_t> Shape::dim_vector() const {
  return dims_;
}

int64_t Shape::stride(int64_t axis) const {
  if (axis < 0) {
    axis += rank();
  }

  ARGUMENT_CHECK(0 <= axis && axis < rank(),
                 "the axis is out of rang: [0, " << rank() << "]");

  return strides_[axis];
}

// enlarge current shape's rank to new rank by add 1.
Shape Shape::enlarge(int64_t rank) const {
  ARGUMENT_CHECK(rank >= this->rank() && rank <= MAX_SHAPE_RANK, "rank out of range");

  auto new_dims = dims_;

  while (new_dims.size() < rank) {
    new_dims.insert(new_dims.begin(), 1);
  }

  return Shape(new_dims);
}

std::string Shape::to_str() const {
  std::stringstream ss;

  ss << "[";

  for (int64_t i = 0; i < this->rank() - 1; ++i) {
    ss << std::to_string(this->dim(i)) << ",";
  }

  if (this->rank() > 0) {
    ss << std::to_string(this->dim(this->rank() - 1));
  }

  ss << "]";

  return ss.str();
}

}