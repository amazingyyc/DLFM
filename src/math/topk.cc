#include "math/topk.h"

namespace dlfm {
namespace math {

template <typename T, bool largest>
struct Element {
  T val;
  int64_t idx;

  Element(T v, int64_t i): val(v), idx(i) {
  }

  bool operator < (const Element &other) const {
    if (largest) {
      return val > other.val;
    } else {
      return val < other.val;
    }
  }
};

// For last dimension topk.
template <typename T, bool largest>
void topk_tail_block_impl(T *x, T *y, int64_t *indices, int64_t n, int64_t k, bool sorted) {
  std::priority_queue<Element<T, largest>> p_queue;

  for (int64_t i = 0; i < n; ++i) {
    Element<T, largest> e(x[i], i);

    if (p_queue.size() < k) {
      p_queue.push(std::move(e));
    } else {
      const auto &top = p_queue.top();

      if (e < top) {
        p_queue.pop();
        p_queue.push(std::move(e));
      }
    }
  }

  std::vector<Element<T, largest>> vec;
  vec.reserve(p_queue.size());

  while (!p_queue.empty()) {
    vec.emplace_back(p_queue.top());
    p_queue.pop();
  }

  if (sorted) {
    std::sort(vec.begin(), vec.end(), [](const Element<T, largest> &e1, const Element<T, largest> &e2) {
      return e1 < e2;
    });
  }

  for (int64_t i = 0; i < k; ++i) {
    y[i] = vec[i].val;
    indices[i] = vec[i].idx;
  }
}

template <typename T, bool largest>
void topk_tail_impl(const Tensor &x, Tensor &y, Tensor &indices, int64_t k, bool sorted) {
  int64_t col = x.shape()[-1];
  int64_t row = x.shape().size() / col;

  auto eigen_device = y.eigen_device();

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (row + num_threads - 1) / num_threads;

  num_threads = (row + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  auto block = [](T *x, T *y, int64_t *indices, int64_t start, int64_t end, int64_t col, int64_t k, bool sorted) {
    for (int64_t r = start; r < end; ++r) {
      topk_tail_block_impl<T, largest>(x + r * col, y + r * k, indices + r * k, col, k, sorted);
    }
  };

  T *xp = x.data<T>();
  T *yp = y.data<T>();
  int64_t *indicesp = indices.data<int64_t>();

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t end = (std::min)(start + block_size, row);

    eigen_device->enqueue_with_barrier(
      &barrier,
      block,
      xp,
      yp,
      indicesp,
      start,
      end,
      col,
      k,
      sorted);
  }

  barrier.Wait();
}

void topk(const Tensor &x, Tensor &y, Tensor &indices, int64_t k, int64_t axis, bool largest, bool sorted) {
  ARGUMENT_CHECK(-1 == axis || axis == x.shape().ndims() - 1, "topk only support last dimension");
  ARGUMENT_CHECK(indices.element_type().is<int64_t>(), "topk need indices is int64_t");

  if (x.element_type().is<float>()) {
    if (largest) {
      topk_tail_impl<float, true>(x, y, indices, k, sorted);
    } else {
      topk_tail_impl<float, false>(x, y, indices, k, sorted);
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}
}