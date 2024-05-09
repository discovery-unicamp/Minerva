from abc import abstractmethod
import copy
import os
from pathlib import Path
from typing import Any, List, Dict
from uuid import uuid4
from time import time
import traceback
import sys
from lightning.pytorch.core.mixins import HyperparametersMixin
import pkg_resources
import yaml
from datetime import datetime

import git
import sys
import platform
from functools import cached_property


class Pipeline(HyperparametersMixin):
    def __init__(
        self,
        cwd: Path | str = None,
        ignore: str | List[str] = None,
        cache_result: bool = False,
        save_run_status: bool = False,
    ):
        self._initialize_vars()
        self._pipeline_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + str(
            uuid4().hex
        )
        self._cache_result = cache_result
        self._save_run_status = save_run_status
        self._cached_run_status = []
        # Current Working Dir
        self._cwd = cwd or Path.cwd()
        if not isinstance(self._cwd, Path):
            self._cwd = Path(self._cwd)
        self._cwd = self._cwd.absolute()
        # Hyperparameters
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

    @property
    def pipeline_id(self):
        return self._pipeline_id

    @property
    def result(self) -> Any:
        return self._result

    @property
    def working_dir(self):
        return self._cwd

    @property
    def pipeline_info(self):
        return {
            "class_name": self.__class__.__name__,
            "created_time": self._created_at,
            "pipeline_id": self.pipeline_id,
            "working_dir": str(self._cwd),
            "run_count": self._run_count,
        }

    @property
    def config(self):
        params = dict(self.hparams)
        return dict(params)

    @cached_property
    def system_info(self) -> Dict[str, Any]:
        d = dict()

        # ---------- Add host information ----------
        d["host"] = {
            "architecture": " ".join(platform.architecture()),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cores": os.cpu_count(),
        }

        # ---------- Add git information ----------
        try:
            repo = git.Repo(search_parent_directories=True)
            git_info = {
                "branch": repo.active_branch.name,
                "commit": repo.head.commit.hexsha,
                "commit_time": repo.head.commit.committed_date,
                "commit_message": repo.head.commit.message,
                "author": repo.head.commit.author.name,
                "author_email": repo.head.commit.author.email,
            }
        except:
            git_info = {}
        d["git"] = git_info

        # ---------- Add python information ----------
        packages = [
            f"{pkg.project_name}=={pkg.version}"
            for pkg in pkg_resources.working_set
        ]
        d["python"] = {
            "pip_packages": packages,
            "version": sys.version,
            "implementation": sys.implementation.name,
            "cwd": str(Path.cwd()),
        }
        
        d["cmd"] = f"{sys.executable} {' '.join(sys.argv)}"

        # ---------- Add pipeline information ----------

        # --------------------------------------------
        return d

    @property
    def run_status(self) -> Dict[str, Any]:
        return {
            "status": self._run_status,
            "start_time": self._run_start_time,
            "end_time": self._run_end_time,
            "exception_info": self._run_exception,
            "cached_result": self._result is not None,
        }

    @property
    def full_info(self) -> Dict[str, Any]:
        d = dict()
        d["pipeline"] = self.pipeline_info
        d["config"] = self.config
        d.update(self.system_info)
        d["runs"] = self._cached_run_status
        return d

    @staticmethod
    def clone(other: "Pipeline") -> "Pipeline":
        clone_pipeline = copy.deepcopy(other)
        clone_pipeline._initialize_vars()
        return clone_pipeline

    def _save_pipeline_info(self, path):
        d = self.full_info
        # Save yaml
        with open(path, "w") as f:
            yaml.dump(d, f)

        print(f"Pipeline info saved at: {path}")

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

        if self._save_run_status:
            self._cached_run_status.append(self.run_status)
            self._save_pipeline_info(self._cwd / f"run_{self.pipeline_id}.yaml")

        return result

    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Must be implemented in derived classes")
