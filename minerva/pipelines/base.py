import copy
import os
import platform
import sys
import traceback
from abc import abstractmethod
from datetime import datetime
from functools import cached_property
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import git
import lightning as L
import pkg_resources
import yaml
from lightning.pytorch.core.mixins import HyperparametersMixin  # type: ignore
from lightning.pytorch.utilities import rank_zero_only

from minerva.utils.typing import PathLike


class Pipeline(HyperparametersMixin):
    """Pipelines provide a versatile API for automating tasks efficiently.
    They are runnable objects that keeps track of their parameters, results, and
    status, allowing the reproductibility and traceability of the experiments.

    This is the base class for all pipelines. It provides the basic structure
    for running a pipeline and saving the results and status of the runs.
    Users should inherit from this class and implement the `_run` method.

    Pipelines are clonal objects, meaning that they can be cloned to create
    new pipelines with the same configuration. Cloned pipelines do receive a
    new pipeline_id and run_count.

    Pipelines expose their public API though properties (which are read-only)
    and though the `run` method. Users should not access or modify the internal
    attributes directly. The run method may set desired attributed (hence
    properties), used to be accessed after or during the run. The run method
    may return a result, which can be cached and accessed through the `result`
    property (if the `cache_result` is set to True).
    """

    def __init__(
        self,
        log_dir: Optional[PathLike] = None,
        ignore: Optional[Union[str, List[str]]] = None,
        cache_result: bool = False,
        save_run_status: bool = False,
        seed: Optional[int] = None,
    ):
        """Create a new Pipeline object.

        Parameters
        ----------
        log_dir : PathLike, optional
            The default logging directory where all related pipeline files
            should be saved. By default None (uses current working directory)
        ignore : Union[str, List[str]], optional
            Pipeline __init__ attributes are saved into config attibute. This
            option allows to ignore some attributes from being saved. This is
            quite useful when the attributes are not serializable or very large.
            By default None (save all __init__ attribute values)
        cache_result : bool, optional
            If True, the result of the last execution of `run` method is stored
            at the `result` attribute. This is useful to avoid recomputing the
            same result multiple times. If False, the result is not stored, by
            default False
        save_run_status : bool, optional
            If True, save the status of each run in a YAML file. This file will
            be saved in the working directory with the name
            `run_{pipeline_id}.yaml`. By default False.
        seed : Optional[int], optional
            Seed to be used by the pipeline. If None, a random seed is generated
            and used. By default None.
        """
        self._initialize_vars()
        self.seed = seed or self._generate_seed()
        self._pipeline_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + str(
            uuid4().hex[:8]
        )
        self._cache_result = cache_result
        self._save_run_status = save_run_status
        self._cached_run_status = []
        # Log dir (set as property)
        self.log_dir = log_dir or Path.cwd()
        self.seed = seed
        # Hyperparameters
        ignore = ignore or []
        if isinstance(ignore, str):
            ignore = [ignore]
        ignore.append("ignore")
        self.save_hyperparameters(ignore=ignore)

    def _generate_seed(self) -> int:
        seed = int(time() * 1000000)  # Using microseconds for higher precision
        return seed

    def _initialize_vars(self):
        """Initialize the internal variables of the pipeline. This method is
        used on __init__ and on clone method.
        """
        self._created_at = time()
        self._run_count = 0
        self._run_start_time = None
        self._run_end_time = None
        self._result = None
        self._run_status = "NOT STARTED"
        self._run_exception = None

    @property
    def pipeline_id(self) -> str:
        """Return the ID of the pipeline. This ID is unique for each pipeline
        object and is generated at the creation of the object.

        Returns
        -------
        str
            The pipeline ID
        """
        return self._pipeline_id

    @property
    def result(self) -> Any:
        """Return the cached result of the last run. If the `cache_result` is
        set to False, this property will return None.

        Returns
        -------
        Any
            The result of the last run.
        """
        return self._result

    @property
    def log_dir(self) -> Path:
        """Return the log_dir where everything inside pipeline should be saved.

        Returns
        -------
        Path
            Path to the pipeline's log_dir
        """
        return self._log_dir

    @log_dir.setter
    def log_dir(self, value: PathLike):
        """Set the log_dir.

        Parameters
        ----------
        value : Path | str
            The new working directory path.
        """
        if not isinstance(value, Path):
            value = Path(value)
        self._log_dir = value.absolute()
        # print(f"Log directory set to: {str(self.log_dir)}")

    @property
    def pipeline_info(self) -> Dict[str, Union[str, float, int]]:
        """Return default information about the pipeline. This information
        includes the class name, the creation time, the pipeline ID, the working
        directory, and the number of runs.

        Returns
        -------
        Dict[str, str]
            The dictionary with the pipeline information
        """
        return {
            "class_name": self.__class__.__name__,
            "created_time": self._created_at,
            "pipeline_id": self.pipeline_id,
            "log_dir": str(self.log_dir),
            "run_count": self._run_count,
        }

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration of the pipeline. This configuration includes the
        __init__ attributes of the pipeline, except the ones that are ignored.

        Returns
        -------
        Dict[str, Any]
            The configuration of the pipeline.
        """
        params = dict(self.hparams)
        return dict(params)

    @cached_property
    def system_info(self) -> Dict[str, Any]:
        """System information about the host, the python environment, and the
        git repository (if available).

        Returns
        -------
        Dict[str, Any]
            The dictionary with the system information.
        """
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
            f"{pkg.project_name}=={pkg.version}" for pkg in pkg_resources.working_set
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
        """Status of the last run of the pipeline.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the status of the last run.
        """
        return {
            "status": self._run_status,
            "start_time": self._run_start_time,
            "end_time": self._run_end_time,
            "exception_info": self._run_exception,
            "cached_result": self._result is not None,
        }

    @property
    def full_info(self) -> Dict[str, Any]:
        """Get all information about the pipeline. This includes, the pipeline
        information, the configuration, the system information, and the status
        of the last run.

        Returns
        -------
        Dict[str, Any]
            The dictionary with all information about the pipeline.
        """
        d = dict()
        d["pipeline"] = self.pipeline_info
        d["config"] = self.config
        d.update(self.system_info)
        d["runs"] = self._cached_run_status
        return d

    @staticmethod
    def clone(other: "Pipeline") -> "Pipeline":
        """Clone a pipeline object. This method creates a new pipeline object
        with the same configuration as the original pipeline. The new pipeline
        will have a new pipeline ID and a new run count.

        Parameters
        ----------
        other : Pipeline
            The pipeline object to be cloned.

        Returns
        -------
        Pipeline
            The new pipeline object (deep-copyied)
        """
        clone_pipeline = copy.deepcopy(other)
        clone_pipeline._initialize_vars()
        return clone_pipeline

    @rank_zero_only
    def _save_pipeline_info(self, path: PathLike):
        """Save the pipeline information to a YAML file.

        Parameters
        ----------
        path : PathLike
            The path to save the pipeline information.
        """
        if not self._save_run_status:
            return

        path = Path(path)
        d = self.full_info
        # Save yaml
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f)

        print(f"Pipeline info saved at: {path}")

    def run(self, *args, **kwargs) -> Any:
        """Default entry-point for running the pipeline. This method calls the
        `_run` method, which should be implemented in the derived classes. This
        method handles the status of the run, the caching of the result, and the
        saving of the run status.

        Returns
        -------
        Any
            The result of the run, from the `_run` method.

        Raises
        ------
        Exception
            Raises any exception that occurs during the run.
        """
        seed = L.seed_everything(self.seed, workers=True, verbose=False)
        print(f"** Seed set to: {seed} **")

        self._run_count += 1
        self._run_start_time = time()
        self._run_status = "RUNNING"
        self._result = None
        self._save_pipeline_info(self._log_dir / f"run_{self.pipeline_id}.yaml")

        try:
            result = self._run(*args, **kwargs)
        except KeyboardInterrupt as e:
            self._run_status = "INTERRUPTED"
            raise e
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

        self._cached_run_status.append(self.run_status)
        self._save_pipeline_info(self._log_dir / f"run_{self.pipeline_id}.yaml")

        return result

    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        """Default pipeline method to be implemented in derived classes. This
        implements the pipeline logic.

        Returns
        -------
        Any
            The result of the pipeline run.
        """
        raise NotImplementedError("Must be implemented in derived classes")
