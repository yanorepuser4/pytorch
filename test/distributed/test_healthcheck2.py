import os
from datetime import timedelta
import time
import shutil

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)

from torch._C._distributed_c10d import (
    HealthcheckNCCL,
    FileStore,
)

store_path = "/tmp/healthcheckpath_test"
rank = int(os.environ["RANK"])
world_size = 2

torch.cuda.set_device(rank)

store2 = FileStore(store_path, world_size)

healthcheck = HealthcheckNCCL(
    store=store2,
    rank=rank,
    world_size=world_size,
    local_world_size=1,
    abort_on_error=False,
    interval=timedelta(milliseconds=10),
    timeout=timedelta(seconds=10000),
)
while healthcheck.num_failures == -1:
    time.sleep(0.01)
healthcheck.shutdown()
