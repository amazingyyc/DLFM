#ifndef COST_HELPER_H
#define COST_HELPER_H

#include "basic.h"

namespace dlfm {

struct TimePoint {
  // name
  std::string name_;

  // millisecond_
  long long millisecond_;

  // microseconds
  long long microseconds_;
};

class CostHelper {
private:
  // use a stack to store the time point
  std::stack<TimePoint> time_points_;

  void start_impl(std::string name);
  void end_impl();

  long long get_cur_millisecond();

  long long get_cur_microseconds();

public:
  static void start(std::string name);
  static void end();

};

}

#endif