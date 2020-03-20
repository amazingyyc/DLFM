#ifndef BASIC_H
#define BASIC_H

#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <ostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cfloat>
#include <limits>
#include <random>
#include <functional>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <stack>
#include <locale>
#include <codecvt>
#include <cwctype>
#include <cctype>
#include <atomic>
#include <cstdlib>
#include <utility>
#include <iomanip>

#ifdef __GNUC__
#include <unistd.h>
#endif

// neon
#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include <eigen/Eigen/Dense>
#include <eigen/unsupported/Eigen/CXX11/Tensor>
#include <eigen/unsupported/Eigen/CXX11/ThreadPool>

// PI
#define PI 3.14159265358979323846

#ifdef __GNUC__
#define FILE_SEP "/"
#elif defined(_MSC_VER)
#define FILE_SEP "\\"
#endif

#endif