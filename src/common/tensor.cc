#include "common/tensor.h"

namespace dlfm {

Tensor::Tensor(std::shared_ptr<TensorStorage> storage, size_t offset, Shape shape, ElementType element_type)
  : storage_(storage), offset_(offset), shape_(shape), element_type_(element_type) {
}

Tensor::Tensor()
  : offset_(0), shape_(), element_type_(ElementType::from<UnKnownType>()) {}

std::shared_ptr<Device> Tensor::device() const {
  return storage_->device();
}

// get eigen device
std::shared_ptr<Eigen::ThreadPoolDevice> Tensor::eigen_device() const {
  return storage_->device()->eigen_device();
}

#ifdef HAS_NNPACK
pthreadpool_t Tensor::nnpack_threadpool() {
  return storage_->device()->nnpack_threadpool();
}
#endif

size_t Tensor::offset() const {
  return offset_;
}

const Shape &Tensor::shape() const {
  return shape_;
}

const ElementType &Tensor::element_type() const {
  return element_type_;
}

void *Tensor::ptr() {
  return ((uint8_t *)storage_->ptr()) + offset_;
}

void *Tensor::ptr() const {
  return ((uint8_t *)storage_->ptr()) + offset_;
}

bool Tensor::is_scalar() const {
  return shape_.is_scalar();
}

int64_t Tensor::ndims() const {
  return shape_.ndims();
}

int64_t Tensor::rank() const {
  return shape_.rank();
}

int64_t Tensor::size() const {
  return shape_.size();
}

size_t Tensor::num_bytes() const {
  return element_type_.byte_width() * ((size_t)size());
}

int64_t Tensor::dim(int64_t axis) const {
  return shape_.dim(axis);
}

int64_t Tensor::stride(int64_t axis) const {
  return shape_.stride(axis);
}

}