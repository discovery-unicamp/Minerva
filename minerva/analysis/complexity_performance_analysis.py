from minerva.analysis.model_analysis import _ModelAnalysis
from thop import profile
from codecarbon import EmissionsTracker
import lightning as L
from typing import Optional, Tuple
import torch
from minerva.data.data_module_tools import get_full_data_split
from minerva.utils.typing import PathLike


class ComplexityPerformanceAnalysis(_ModelAnalysis):
    """Perform a complexity/performance analysis on a model using custom data.
    The values computed are the KWh (kilowatts per hour) consumed, MACs (multiply-
    accumulative operations) employed, and the number of parameters in the model.
    The results are returned in a dictionary and, if provided, saved in the `path`
    directory or returned in the compute function.
    If necessary, random data can be employed.
    """

    def __init__(
        self, path: Optional[PathLike] = None, custom_input_size: Optional[Tuple] = None
    ):
        super().__init__()
        self._path = path
        self._custom_input_size = custom_input_size
        """Compute the complexity/performance analysis.

        Parameters
        ----------
        path : Optional[PathLike], optional
            Path to save the results of the analysis, by default None.
        custom_input_size : Optional[Tuple], optional
            Custom input size for the evaluation data, by default None.
            If None, the evaluation data is obtained from the data module.
        """

    def compute(self, model: L.LightningModule, data: L.LightningDataModule):
        # Establishing the data to be used
        if self._custom_input_size:
            evaluation_data = torch.rand(self._custom_input_size)
        else:
            evaluation_data = get_full_data_split(data, "predict")[0]
            evaluation_data = torch.tensor(evaluation_data, dtype=torch.float32)
        # Computing MACs, parameters, and energy consumption
        macs, params = profile(model, inputs=(evaluation_data,))
        carbonTracker = EmissionsTracker(
            project_name="basic_measurement", measure_power_secs=10, save_to_file=False
        )
        try:
            carbonTracker.start_task("measure_inference")
            _ = model(evaluation_data)
            model_emissions = carbonTracker.stop_task()
        finally:
            carbonTracker.stop()
        # Saving the results
        result = {
            "KWh": model_emissions.energy_consumed,
            "Macs": int(macs),
            "Params": int(params),
        }
        return result
