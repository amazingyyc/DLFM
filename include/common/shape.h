#ifndef SHAPE_H
#define SHAPE_H

#include "basic.h"

namespace dlfm {

#define MAX_SHAPE_RANK 4

class Shape {
private:
  std::vector<int64_t> dims_;
  std::vector<int64_t> strides_;

private:
  void update_strides();

public:
  explicit Shape();
  explicit Shape(std::vector<int64_t> dims);

  Shape &operator=(const Shape &);

  bool operator==(const Shape &) const;
  bool operator!=(const Shape &) const;

  int64_t operator[](int64_t) const;

  void reset(std::vector<int64_t> dims);

  bool is_scalar() const;

  int64_t ndims() const;

  int64_t rank() const;

  int64_t size() const;

  int64_t dim(int64_t) const;

  std::vector<int64_t> dim_vector() const;

  int64_t stride(int64_t) const;

  std::string to_str() const;
};

}

#endif