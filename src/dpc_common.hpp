//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include <stdlib.h>
#include <exception>

#include <CL/sycl.hpp>

  class NEOGPUDeviceSelector : public cl::sycl::device_selector {
  public:
    int operator()(const cl::sycl::device &Device) const override {
      using namespace cl::sycl::info;

      const std::string DeviceName = Device.get_info<device::name>();
      const std::string DeviceVendor = Device.get_info<device::vendor>();
 	    std::cout << DeviceName << " " << Device.is_gpu() << std::endl;
      return Device.is_cpu() && (DeviceName.find("Corce") != std::string::npos);
    }
  };
  
namespace dpc_common {
// This exception handler with catch async exceptions
static auto exception_handler = [](cl::sycl::exception_list eList) {
  for (std::exception_ptr const &e : eList) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

// The TimeInterval is a simple RAII class.
// Construct the timer at the point you want to start timing.
// Use the Elapsed() method to return time since construction.

class TimeInterval {
 public:
  TimeInterval() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

 private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

};  // namespace dpc_common
