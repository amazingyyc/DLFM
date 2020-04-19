#include "common/exception.h"
#include "common/cost_helper.h"

namespace dlfm {

CostHelper default_cost_helper;


long long CostHelper::get_cur_millisecond() {
  auto time_now = std::chrono::system_clock::now();
  auto duration_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_now.time_since_epoch());
  return duration_in_ms.count();
}

long long CostHelper::get_cur_microseconds() {
  auto time_now = std::chrono::system_clock::now();
  auto duration_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(time_now.time_since_epoch());
  return duration_in_ms.count();
}

void CostHelper::start_impl(std::string name) {
  TimePoint point;
  point.name_ = name;
  point.millisecond_  = get_cur_millisecond();
  point.microseconds_ = get_cur_microseconds();

  time_points_.push(point);
}

void CostHelper::end_impl() {
  auto cur_millisecond = get_cur_millisecond();
  auto cur_microseconds = get_cur_microseconds();

  ARGUMENT_CHECK(!time_points_.empty(), "time point is empty");

  auto prev_point = time_points_.top();
  time_points_.pop();

  std::cout << "[" << prev_point.name_ << "] => microseconds:"
            << (cur_microseconds - prev_point.microseconds_) << ", millisecond:"
            << (cur_millisecond - prev_point.millisecond_) << std::endl;
}

void CostHelper::start(std::string name) {
  default_cost_helper.start_impl(name);
}

void CostHelper::end() {
  default_cost_helper.end_impl();
}

}