#include <sys/socket.h>
#include <exception>
#include <future>

#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/Healthcheck.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d {

Healthcheck::Healthcheck(
    bool abortOnError,
    std::chrono::milliseconds interval,
    std::chrono::milliseconds timeout)
    : abortOnError_(abortOnError), interval_(interval), timeout_(timeout) {
  worker_ = std::async(std::launch::async, [this]() {
    try {
      runLoop();
    } catch (const std::exception& e) {
      C10D_ERROR("Healthcheck thread failed: {}", e.what());
    } catch (...) {
      C10D_ERROR("Healthcheck thread failed with unknown exception");
    }
  });
}

void Healthcheck::runLoop() {
  C10D_ERROR("Healthcheck setup...");
  for (int i = 0; i < 2; i++) {
    setup(i);
  }
  C10D_ERROR("Healthcheck setup complete!");

  while (true) {
    C10D_ERROR("Running healthchecks...");

    std::vector<std::future<void>> futures;
    futures.reserve(2);

    for (int i = 0; i < 1; i++) {
      futures.emplace_back(std::async(
          std::launch::async,
          [](Healthcheck* self, int i) {
            self->runHealthcheck(i);
            C10D_ERROR("Worker exit {}", i);
          },
          this,
          i));
    }

    // calculate deadline for the futures
    std::chrono::time_point<std::chrono::system_clock> deadline =
        std::chrono::system_clock::now() + timeout_;

    int failures = 0;

    // wait for futures to complete
    for (auto& future : futures) {
      C10D_ERROR("waiting...");
      auto status = future.wait_until(deadline);
      C10D_ERROR("Healthcheck returned");
      if (status == std::future_status::timeout) {
        failures += 1;
        C10D_ERROR("Healthcheck timed out");
        continue;
      }
      TORCH_INTERNAL_ASSERT(status == std::future_status::ready);

      try {
        future.get();
        C10D_ERROR("Healthcheck passed");
      } catch (const std::exception& e) {
        C10D_ERROR("Healthcheck failed: {}", e.what());
        failures += 1;
        continue;
      } catch (...) {
        C10D_ERROR("Healthcheck failed with unknown exception");
        failures += 1;
        continue;
      }
    }

    C10D_ERROR("Healthcheck had {} failures", failures);
    numFailures_ = failures;
    if (failures == 2) {
      C10D_ERROR("Current host identified as problematic!");
      if (abortOnError_) {
        std::abort();
      }
    }

    // wait for interval
    {
      std::unique_lock lock{shutdownM_};
      shutdownCv_.wait_for(lock, interval_);
      if (shutdown_) {
        break;
      }
    }
  }
}

void Healthcheck::shutdown() {
  {
    std::unique_lock lock{shutdownM_};
    shutdown_ = true;
  }
  shutdownCv_.notify_all();

  worker_.get();
}

} // namespace c10d
