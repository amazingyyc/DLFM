#ifndef TENSOR_H
#define TENSOR_H

#include "basic.h"
#include "element_type.h"
#include "shape.h"
#include "tensor_storage.h"

namespace dlfm {

class Tensor {
 private:
  // store the memory
  std::shared_ptr<TensorStorage> storage_;

  // memory offset
  size_t offset_;

  // shape
  Shape shape_;

  // element type
  ElementType element_type_;

  explicit Tensor(std::shared_ptr<TensorStorage>, size_t, Shape, ElementType);

 public:
  Tensor();

  std::shared_ptr<Device> device() const;

  std::shared_ptr<Eigen::ThreadPoolDevice> eigen_device() const;

#ifdef HAS_NNPACK
  pthreadpool_t nnpack_threadpool();
#endif

  size_t offset() const;

  const Shape &shape() const;

  const ElementType &element_type() const;

  void *ptr();
  void *ptr() const;

  template <typename T>
  T *data() {
    return (T *)ptr();
  }

  template <typename T>
  T *data() const {
    return (T *)ptr();
  }

  bool is_scalar() const;

  int64_t ndims() const;

  int64_t rank() const;

  int64_t size() const;

  size_t num_bytes() const;

  int64_t dim(int64_t) const;

  int64_t stride(int64_t) const;

 public:
  static Tensor create(std::vector<int64_t> dims,
                       ElementType type = ElementType::from<float>());

  static Tensor create(Shape &shape,
                       ElementType type = ElementType::from<float>());

  static Tensor create_from(void *ptr, std::vector<int64_t> dims,
                            ElementType type);

  static Tensor create_from(void *ptr, Shape &shape, ElementType type);

  static Tensor zeros(std::vector<int64_t> dims,
                      ElementType type = ElementType::from<float>());

  static Tensor ones(std::vector<int64_t> dims,
                     ElementType type = ElementType::from<float>());

 public:
  Tensor initialize_from_file(std::string path);

  // operator override
  Tensor operator+=(const Tensor &other);
  Tensor operator-=(const Tensor &other);
  Tensor operator*=(const Tensor &other);
  Tensor operator/=(const Tensor &other);

  Tensor operator+(const Tensor &other);
  Tensor operator-(const Tensor &other);
  Tensor operator*(const Tensor &other);
  Tensor operator/(const Tensor &other);

  // operator override
  Tensor operator+=(float);
  Tensor operator-=(float);
  Tensor operator*=(float);
  Tensor operator/=(float);

  Tensor operator+(float);
  Tensor operator-(float);
  Tensor operator*(float);
  Tensor operator/(float);

  Tensor reshape(std::vector<int64_t>);

  Tensor reshape(std::vector<int64_t>) const;

  Tensor reshape(Shape &);

  // shape and type like this tensor
  Tensor like();

  Tensor clone();

  Tensor mean(std::vector<int64_t> axis = {}, bool keep_dims = false);

  Tensor sum(std::vector<int64_t> axis = {}, bool keep_dims = false);

  Tensor var(std::vector<int64_t> axis = {}, bool keep_dims = false, bool unbiased = true);

  Tensor std(std::vector<int64_t> axis = {}, bool keep_dims = false, bool unbiased = true);

  // set all element of this tensor to be value.
  Tensor fill(float);

  Tensor relu(bool);

  Tensor sigmoid(bool);

  Tensor tanh(bool);

  Tensor sqrt(bool);

  Tensor square(bool);

  Tensor cast(ElementType to_type);

  Tensor slice(std::vector<int64_t> offsets, std::vector<int64_t> extents);

  Tensor reverse(std::vector<bool>);

  Tensor transpose(std::vector<size_t> axis);

  // pad like pytorh torch.nn.functional.pad(input, pad, mode='constant', value=0)
  Tensor pad(std::vector<size_t> paddings);

  Tensor reflection_pad2d(size_t padding);
  Tensor reflection_pad2d(std::vector<size_t> paddings);

  Tensor cat(const Tensor &, int64_t axis);

  // kernel, stride, padding size both 2
  // corresponding pytorch
  Tensor max_pooling2d(std::vector<size_t> kernel_size, std::vector<size_t> stride,
                     std::vector<size_t> padding, bool ceil_mode = false);

  Tensor upsample2d(float scale_factor, std::string mode="nearest", bool align_corners = false);

  Tensor matmul(const Tensor &y, bool transpose_a = false, bool transpose_b = false);

  // conv2d
  Tensor conv2d(const Tensor &weight, const Tensor &bias,
                std::vector<size_t> stride, std::vector<size_t> padding);

  // transpose conv2d
  Tensor conv_transpose2d(const Tensor &weight, const Tensor &bias,
                          std::vector<size_t> stride,
                          std::vector<size_t> padding,
                          std::vector<size_t> out_padding);

  Tensor instance_norm2d(float eps = 1e-05);
  Tensor instance_norm2d(Tensor &scale, Tensor &shift, float eps = 1e-05);

  ////////////////////////////////////////////////////////////////////////////////////

  template <typename T>
  std::ostream& pretty_print(std::ostream &os) const {
    auto ndims = shape_.ndims();

    if (1 == ndims) {
      os << "vector:\n";

      for (int64_t i = 0; i < shape_.size(); ++i) {
        os << std::setw(4) << data<T>()[i] << " ";
      }

      os << "\n";
    } else if (2 == ndims) {
      os << "matrix:\n";

      auto col = shape_[-1];
      auto row = shape_[-2];

      for (int64_t r = 0; r < row; ++r) {
        for (int64_t c = 0; c < col; ++c) {
          os << std::setw(4) << data<T>()[r * col + c] << " ";
        }

        os << "\n";
      }
    } else {
      // print like matrix
      auto col = shape_[-1];
      auto row = shape_[-2];
      auto n = shape_.size() / row / col;

      for (int64_t i = 0; i < n; ++i) {
        os << "matrix " << i << ":\n";

        for (int64_t r = 0; r < row; ++r) {
          for (int64_t c = 0; c < col; ++c) {
            os << std::setw(4) << data<T>()[i * row * col +  r * col + c] << " ";
          }

          os << "\n";
        }

        os << "\n";
      }
    }

    os << "shape:" << shape_.to_str() << "\n";

    return os;
  }
};

// operator override
std::ostream& operator<<(std::ostream& os, const Tensor &t);

}  // namespace dlfm

#endif