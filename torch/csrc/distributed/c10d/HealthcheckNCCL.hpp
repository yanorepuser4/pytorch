#pragma once

#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Export.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Healthcheck.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {

class TORCH_API HealthcheckNCCL : public Healthcheck {
 public:
  HealthcheckNCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int worldSize,
      int localWorldSize,
      bool abortOnError = false,
      std::chrono::milliseconds interval = std::chrono::seconds(10),
      std::chrono::milliseconds timeout = std::chrono::seconds(10));

 protected:
  void setup(int side) override;
  void runHealthcheck(int side) override;

 private:
  const int rank_;
  const int worldSize_;
  const int localWorldSize_;
  const c10::DeviceIndex deviceIndex_;

  const c10::intrusive_ptr<Store> store_;
  std::vector<at::cuda::CUDAStream> streams_;
  std::vector<c10::intrusive_ptr<::c10d::Backend>> processGroups_;
};

} // namespace c10d
