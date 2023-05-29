import logging
from typing import Any

from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback

from omegaconf import DictConfig


class MyLogJobReturnCallback(Callback):
    """Log the job's return value or error upon job end"""

    def __init__(self) -> None:
        self.log = logging.getLogger("__main__")
        # self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        with open(
            "/mmfs1/gscratch/stf/abhinavp/573-SADTech/src/LOOKHERE.err", "a"
        ) as f:
            if job_return.status == JobStatus.COMPLETED:
                msg = f"Succeeded with return value: {job_return.return_value}"
                self.log.info(msg)
                print(msg, file=f, flush=True)
            elif job_return.status == JobStatus.FAILED:
                print(job_return._return_value, file=f, flush=True)
                self.log.error("", exc_info=job_return._return_value, stack_info=True)
            else:
                print(job_return._return_value, file=f, flush=True)
                self.log.error(
                    "Status unknown. This should never happen.", stack_info=True
                )
