import shutil
import time
from datetime import timedelta

import torch
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


class HealthcheckTest(MultiProcessTestCase):
    store_path: str = "/tmp/test_healthcheck.filestore"

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self) -> None:
        shutil.rmtree(self.store_path, ignore_errors=True)

        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    def test_healthcheck_nccl(self) -> None:
        from torch._C._distributed_c10d import FileStore, HealthcheckNCCL

        torch.cuda.set_device(self.rank)

        store2 = FileStore(self.store_path, self.world_size)

        healthcheck = HealthcheckNCCL(
            store=store2,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=1,
            abort_on_error=False,
            interval=timedelta(milliseconds=10),
            timeout=timedelta(seconds=10),
        )
        while healthcheck.num_failures == -1:
            time.sleep(0.01)
        healthcheck.shutdown()


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
