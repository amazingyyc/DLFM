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
  static Tensor create(const std::vector<int64_t> &dims, ElementType type = ElementType::from<float>());

  static Tensor create(const Shape &shape, ElementType type = ElementType::from<float>());

  static Tensor create_from(void *ptr, const std::vector<int64_t> &dims, ElementType type);

  static Tensor create_from(void *ptr, const Shape &shape, ElementType type);

  static Tensor zeros(const std::vector<int64_t> &dims, ElementType type = ElementType::from<float>());

  static Tensor ones(const std::vector<int64_t> &dims, ElementType type = ElementType::from<float>());

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

  Tensor operator[](int64_t idx);

  Tensor reshape(const std::vector<int64_t>&);

  Tensor reshape(const std::vector<int64_t>&) const;

  Tensor reshape(const Shape&);

  Tensor view(std::vector<int64_t>);

  Tensor unsqueeze(size_t axis);
  Tensor squeeze(size_t axis);

  // shape and type like this tensor
  Tensor like();

  Tensor clone();

  Tensor mean(int64_t axis, bool keep_dims = false);
  Tensor mean(std::vector<int64_t> axis = {}, bool keep_dims = false);

  Tensor sum(std::vector<int64_t> axis = {}, bool keep_dims = false);

  // get variance for input, only suport one axis.
  Tensor var(int64_t axis, const Tensor &mean, bool unbiased = true);
  Tensor var(int64_t axis, bool keep_dims = false, bool unbiased = true);

  Tensor std(int64_t axis, const Tensor &mean, bool unbiased = true);
  Tensor std(int64_t axis, bool keep_dims = false, bool unbiased = true);

  Tensor softmax(int64_t axis);

  Tensor clamp(float min, float max, bool in_place=false);

  // set all element of this tensor to be value.
  Tensor fill(float);

  Tensor relu(bool);

  Tensor relu6(bool);

  Tensor prelu(const Tensor &w, bool);

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

  // (this - mean) / std
  Tensor normalize(Tensor mean, Tensor std, bool in_place=false);

  // (this * std) + mean
  Tensor denormalize(Tensor mean, Tensor std, bool in_place=false);

  // kernel, stride, padding size both 2
  // corresponding pytorch
  Tensor max_pooling2d(std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding, bool ceil_mode = false);

  // return max and indices
  std::vector<Tensor> max_pooling2d_with_indices(std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding, bool ceil_mode = false);

  // pytorch max_unpooling2d
  Tensor max_unpooling2d(const Tensor &indices, std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding);

  // kernel, stride, padding size both 2
  Tensor avg_pooling2d(size_t kernel_size, size_t stride, size_t padding = 0, bool ceil_mode = false);
  Tensor avg_pooling2d(std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding, bool ceil_mode = false);

  Tensor adaptive_avg_pooling2d(size_t size);
  Tensor adaptive_avg_pooling2d(std::vector<size_t> size);

  Tensor adaptive_max_pooling2d(size_t size);
  Tensor adaptive_max_pooling2d(std::vector<size_t> size);

  Tensor upsample2d(float scale_factor, std::string mode="nearest", bool align_corners = false);

  Tensor interpolate2d(std::vector<int64_t> output_size, std::string mode="nearest", bool align_corners = false);

  Tensor pixel_shuffle(int64_t upscale_factor);

  Tensor matmul(const Tensor &y, bool transpose_a = false, bool transpose_b = false);

  // conv2d
  Tensor conv2d(const Tensor &weight, const Tensor &bias, std::vector<size_t> stride, std::vector<size_t> padding, size_t groups = 1);

  // transpose conv2d
  Tensor conv_transpose2d(const Tensor &weight, const Tensor &bias, std::vector<size_t> stride, std::vector<size_t> padding, std::vector<size_t> out_padding);

  Tensor instance_norm2d(float eps = 1e-05);
  Tensor instance_norm2d(const Tensor &scale, const Tensor &shift, float eps = 1e-05);

  Tensor batch_norm2d(const Tensor &mean, const Tensor &var, const Tensor &scale, const Tensor &shift, float eps);

  // special for img
  Tensor img_mask(const Tensor &mask, const Tensor &val);

  //-----------------------------------------------------------------------------------------------------------------------------------------
  template <typename T>
  std::ostream& pretty_print(std::ostream &os) const {
    int64_t show_count = 50;

    auto ndims = shape_.ndims();

    if (1 == ndims) {
      os << "vector:\n";

      if (shape_.size() > 2 * show_count) {
        for (int64_t i = 0; i < shape_.size() && i < show_count; ++i) {
          os << std::setw(4) << data<T>()[i] << " ";
        }

        os << "...";

        for (int64_t i = shape_.size() - show_count; i < shape_.size(); ++i) {
          os << std::setw(4) << data<T>()[i] << " ";
        }
      } else {
        for (int64_t i = 0; i < shape_.size(); ++i) {
          os << std::setw(4) << data<T>()[i] << " ";
        }
      }

      os << "\n";
    } else if (2 == ndims) {
      os << "matrix:\n";

      auto col = shape_[-1];
      auto row = shape_[-2];

      for (int64_t r = 0; r < row; ++r) {
        if (col < show_count * 2) {
          for (int64_t c = 0; c < col; ++c) {
            os << std::setw(4) << data<T>()[r * col + c] << " ";
          }
        } else {
          for (int64_t c = 0; c < col && c < show_count; ++c) {
            os << std::setw(4) << data<T>()[r * col + c] << " ";
          }

          os << "...";

          for (int64_t c = col - show_count; c < col; ++c) {
            os << std::setw(4) << data<T>()[r * col + c] << " ";
          }
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
          if (col < show_count * 2) {
            for (int64_t c = 0; c < col; ++c) {
              os << std::setw(4) << data<T>()[i * row * col + r * col + c] << " ";
            }
          } else {
            for (int64_t c = 0; c < col && c < show_count; ++c) {
              os << std::setw(4) << data<T>()[i * row * col + r * col + c] << " ";
            }

            os << "...";

            for (int64_t c = col - show_count; c < col; ++c) {
              os << std::setw(4) << data<T>()[i * row * col + r * col + c] << " ";
            }
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

// operator override
Tensor operator+(float, const Tensor&);
Tensor operator-(float, const Tensor&);
Tensor operator*(float, const Tensor&);
Tensor operator/(float, const Tensor&);


}  // namespace dlfm

#endif