
from abc import abstractmethod
import copy
from pathlib import Path
from lightning.pytorch.core.mixins import HyperparametersMixin
from typing import Any, List, Dict, Tuple
from uuid import uuid4
from time import time
import traceback
import sys
from jsonargparse import CLI


class Pipeline(HyperparametersMixin):
    def __init__(
        self,
        cwd: Path | str = None,
        ignore: str | List[str] = None,
        cache_result: bool = False,
    ):
        self._initialize_vars()
        self.pipeline_id = str(uuid4().hex)
        self._cache_result = cache_result

        self._cwd = cwd or Path.cwd()
        if not isinstance(self._cwd, Path):
            self._cwd = Path(self._cwd)
        self._cwd = self._cwd.absolute()

        ignore = ignore or []
        if isinstance(ignore, str):
            ignore = [ignore]
        ignore.append("ignore")

        self.save_hyperparameters(ignore=ignore)

    def _initialize_vars(self):
        self._created_at = time()
        self._run_count = 0
        self._run_start_time = None
        self._run_end_time = None
        self._result = None
        self._run_status = "NOT STARTED"
        self._run_exception = None

    def _run(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def clone(other: "Pipeline") -> "Pipeline":
        clone_pipeline = copy.deepcopy(other)
        clone_pipeline._initialize_vars()
        return clone_pipeline

    @abstractmethod
    def run(self, *args, **kwargs):
        self._run_count += 1
        self._run_start_time = time()
        self._run_status = "RUNNING"
        self._result = None

        try:
            result = self._run(*args, **kwargs)
        except Exception as e:
            self._run_status = "FAILED"
            exception = "".join(traceback.format_exception(*sys.exc_info()))
            self._run_exception = exception
            raise e
        finally:
            self._run_end_time = time()

        self._run_status = "SUCCESS"

        if self._cache_result:
            self._result = result

        return result

    @property
    def config(self):
        params = self.hparams
        return dict(params)

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "status": self._run_status,
            "working_dir": str(self._cwd),
            "id": self.pipeline_id,
            "count": self._run_count,
            "created": self._created_at,
            "start_time": self._run_start_time,
            "end_time": self._run_end_time,
            "exception_info": self._run_exception,
            "cached_result": self._result is not None,
        }

    @property
    def result(self) -> Any:
        return self._result

    @property
    def working_dir(self):
        return self._cwd
