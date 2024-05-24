#include <torch/csrc/distributed/c10d/HealthcheckNCCL.hpp>

#include <fmt/format.h>

#include <ATen/ATen.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Healthcheck.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d {

HealthcheckNCCL::HealthcheckNCCL(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int worldSize,
    int localWorldSize,
    bool abortOnError,
    std::chrono::milliseconds interval,
    std::chrono::milliseconds timeout)
    : Healthcheck(abortOnError, interval, timeout),
      rank_(rank),
      worldSize_(worldSize),
      localWorldSize_(localWorldSize),
      deviceIndex_(c10::cuda::current_device()),
      store_(store) {
  if (worldSize % localWorldSize != 0) {
    throw std::runtime_error(
        "World size must be divisible by local world size");
  }
  if (rank >= worldSize) {
    throw std::runtime_error("Rank must be less than world size");
  }
  if (worldSize / localWorldSize < 2) {
    throw std::runtime_error("At least two hosts are required");
  }

  streams_.reserve(2);
  processGroups_.reserve(2);
}

void HealthcheckNCCL::setup(int side) {
  auto hostRank = rank_ / localWorldSize_;
  auto hostCount = worldSize_ / localWorldSize_;

  auto group = (hostRank + side) % hostCount / 2;
  auto groupSize = 2 * localWorldSize_;
  auto groupRank = rank_ % groupSize;

  // C10D_ERROR("store is {}", store_);

  auto storePrefix = fmt::format("/healthcheck/{}/{}", side, group);
  auto store = c10::make_intrusive<PrefixStore>(storePrefix, store_);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  streams_.emplace_back(c10::cuda::getStreamFromExternal(stream, deviceIndex_));

  C10D_ERROR(
      "Creating process group for side side={}, group={}, rank={}, size={}, store={}",
      side,
      group,
      groupRank,
      groupSize,
      storePrefix);

  processGroups_.emplace_back(
      c10::make_intrusive<ProcessGroupNCCL>(store, groupRank, groupSize));
}

void HealthcheckNCCL::runHealthcheck(int side) {
  auto device = deviceIndex_;
  C10D_ERROR("running healthcheck side={} device={}", side, device);

  at::cuda::setCurrentCUDAStream(streams_.at(side));
  auto& pg = processGroups_.at(side);

  at::Tensor t = at::ones(
      {1},
      at::TensorOptions{}
          .device(at::Device(c10::DeviceType::CUDA, device))
          .dtype(at::kFloat));
  std::vector<at::Tensor> tensors{t};

  C10D_ERROR("allreduce side={}", side);

  auto work = pg->allreduce(tensors);
  work->wait(timeout_);

  C10D_ERROR("waited side={}", side);

  if (t.item().to<double>() != 2.0 * localWorldSize_) {
    throw std::runtime_error(
        "Health check all reduce returned invalid results");
  }

  C10D_ERROR("success side={}", side);
}

} // namespace c10d
