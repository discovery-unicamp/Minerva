from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
import torch.nn as nn

from minerva.engines.engine import _Engine


class PatchInferencer(L.LightningModule):
    """Inference in patches for models

    This class provides utility methods for performing inference in patches
    """

    def __init__(
        self,
        model: L.LightningModule,
        input_shape: Tuple,
        output_shape: Optional[Tuple] = None,
        weight_function: Optional[Callable] = None,
        offsets: Optional[List[Tuple]] = None,
        padding: Optional[Dict[str, Any]] = None,
        return_tuple: Optional[int] = None,
    ):
        """Initialize the patch inference auxiliary class

        Parameters
        ----------
        model : L.LightningModule
            Model used in inference.
        input_shape : Tuple
            Expected input shape of the model
        output_shape : Tuple, optional
            Expected output shape of the model. Defaults to input_shape
        weight_function: callable, optional
            Function that receives a tensor shape and returns the weights for each position of a tensor with the given shape
            Useful when regions of the inference present diminishing performance when getting closer to borders, for instance.
        offsets : Tuple, optional
            List of tuples with offsets that determine the shift of the initial position of the patch subdivision
        padding : Dict[str, Any], optional
            Dictionary describing padding strategy. Keys:
                pad: tuple with pad width (int) for each dimension, e.g. (0, 3, 3) when working with a tensor with 3 dimensions
                mode (optional): 'constant', 'reflect', 'replicate' or 'circular'. Defaults to 'constant'.
                value (optional): fill value for 'constant'. Defaults to 0.
        """
        super().__init__()
        self.model = model
        self.patch_inferencer = PatchInferencerEngine(
            input_shape, output_shape, offsets, padding, weight_function, return_tuple
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform Inference in Patches

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor.
        """
        return self.patch_inferencer(self.model, x)

    def _single_step(self, batch: torch.Tensor, batch_idx: int, step_name: str):
        """Perform a single step of the training/validation loop.

        Parameters
        ----------
        batch : torch.Tensor
            The input data.
        batch_idx : int
            The index of the batch.
        step_name : str
            The name of the step, either "train" or "val".

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        x, y = batch
        y_hat = self.forward(x.float())
        loss = self.model._loss_func(y_hat, y.squeeze(1))

        metrics = self.model._compute_metrics(y_hat, y, step_name)
        for metric_name, metric_value in metrics.items():
            self.log(
                metric_name,
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._single_step(batch, batch_idx, "test")


# region _PatchInferencer
class PatchInferencerEngine(_Engine):

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Optional[Tuple[int]] = None,
        offsets: Optional[List[Tuple]] = None,
        padding: Optional[Dict[str, Any]] = None,
        weight_function: Optional[Callable] = None,
        return_tuple: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        model : nn.Module
            The neural network model for inference.
        input_shape : Tuple[int]
            Shape of each patch to process.
        output_shape : Tuple[int], optional
            Expected shape of the model output per patch. Defaults to input_shape.
        padding : dict, optional
            Padding configuration with keys:
                - 'pad': Tuple of padding for each dimension, e.g., (0, 3, 3).
                - 'mode': Padding mode, e.g., 'constant', 'reflect'.
                - 'value': Padding value if mode is 'constant'.
        """
        self.input_shape = (1, *input_shape)
        self.output_shape = (
            (1, *output_shape) if output_shape is not None else self.input_shape
        )

        self.weight_function = weight_function

        if offsets is not None:
            for offset in offsets:
                assert len(input_shape) == len(
                    offset
                ), f"Offset tuple does not match expected size ({len(input_shape)})"
            self.offsets = offsets
        else:
            self.offsets = []

        if padding is not None:
            assert len(input_shape) == len(
                padding["pad"]
            ), f"Pad tuple does not match expected size ({len(input_shape)})"
            self.padding = padding
            self.padding["pad"] = (0, *self.padding["pad"])
        else:
            self.padding = {"pad": tuple([0] * (len(input_shape) + 1))}
        self.return_tuple = return_tuple

    def _reconstruct_patches(
        self,
        patches: torch.Tensor,
        index: Tuple[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rearranges patches to reconstruct area of interest from patches and weights
        """
        reconstruct_shape = np.array(self.output_shape) * np.array(index)
        weight = torch.zeros(tuple(reconstruct_shape), device=patches.device)
        base_weight = (
            self.weight_function(self.output_shape)
            if self.weight_function
            else torch.ones(self.output_shape, device=patches.device)
        )

        reconstruct = torch.zeros(tuple(reconstruct_shape), device=patches.device)
        for patch_index, patch in zip(np.ndindex(index), patches):
            sl = [
                slice(idx * patch_len, (idx + 1) * patch_len, None)
                for idx, patch_len in zip(patch_index, self.output_shape)
            ]
            weight[tuple(sl)] = base_weight
            reconstruct[tuple(sl)] = patch
        return reconstruct, weight

    def _adjust_patches(
        self,
        arrays: List[torch.Tensor],
        ref_shape: Tuple[int],
        offset: Tuple[int],
        pad_value: int = 0,
    ) -> List[torch.Tensor]:
        """
        Pads reconstructed_patches with 'pad_value' to have same shape as the reference shape from the base patch set
        """
        pad_width = []
        sl = []
        ref_shape = list(ref_shape)
        arr_shape = list(arrays[0].shape)
        for idx, lenght, ref in zip([0, *offset], arr_shape, ref_shape):
            if idx > 0:
                sl.append(slice(0, min(lenght, ref), None))
                pad_width = [idx, max(ref - lenght - idx, 0)] + pad_width
            else:
                sl.append(slice(np.abs(idx), min(lenght, ref - idx), None))
                pad_width = [0, max(ref - lenght - idx, 0)] + pad_width
        adjusted = [
            (
                torch.nn.functional.pad(
                    arr[tuple(sl)],
                    pad=tuple(pad_width),
                    mode="constant",
                    value=pad_value,
                )
            )
            for arr in arrays
        ]
        return adjusted

    def _combine_patches(
        self,
        results: List[torch.Tensor],
        offsets: List[Tuple[int]],
        indexes: List[Tuple[int]],
    ) -> torch.Tensor:
        """
        Combination of results
        """
        reconstructed = []
        weights = []
        for patches, offset, shape in zip(results, offsets, indexes):
            reconstruct, weight = self._reconstruct_patches(patches, shape)
            reconstruct, weight = self._adjust_patches(
                [reconstruct, weight], self.ref_shape, offset
            )
            reconstructed.append(reconstruct)
            weights.append(weight)
        reconstructed = torch.stack(reconstructed, dim=0)
        weights = torch.stack(weights, dim=0)
        return torch.sum(reconstructed * weights, dim=0) / torch.sum(weights, dim=0)

    def _extract_patches(
        self, data: torch.Tensor, patch_shape: Tuple[int]
    ) -> Tuple[torch.Tensor, Tuple[int]]:
        """
        Patch extraction method. It will be called once for the base patch set and also for the requested offsets (overlapping patch sets)
        """
        indexes = tuple(np.array(data.shape) // np.array(patch_shape))
        patches = []
        for patch_index in np.ndindex(indexes):
            sl = [
                slice(idx * patch_len, (idx + 1) * patch_len, None)
                for idx, patch_len in zip(patch_index, patch_shape)
            ]
            patches.append(data[tuple(sl)])
        return torch.stack(patches), indexes

    def _compute_output_shape(self, tensor: torch.Tensor) -> Tuple[int]:
        """
        Computes PatchInferencer output shape based on input tensor shape, and model's input and output shapes.
        """
        if self.input_shape == self.output_shape:
            return tensor.shape
        shape = []
        for i, o, t in zip(self.input_shape, self.output_shape, tensor.shape):
            if i != o:
                shape.append(int(t * o // i))
            else:
                shape.append(t)
        return tuple(shape)

    def _compute_base_padding(self, tensor: torch.Tensor):
        """
        Computes the padding for the base patch set based on the input tensor shape and the model's input shape.
        """
        padding = []
        for i, t in zip(self.padding["pad"], tensor.shape[1:]):
            padding.append(max(0, i - t))
        return padding

    def __call__(
        self, model: Union[L.LightningModule, torch.nn.Module], x: torch.Tensor
    ):
        """
        Perform Inference in Patches

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor.
        """
        if len(x.shape) == len(self.input_shape) - 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == len(self.input_shape):
            pass
        else:
            raise RuntimeError("Invalid input shape")

        self.ref_shape = self._compute_output_shape(x)
        offsets = list(self.offsets)
        base = self._compute_base_padding(x)
        offsets.insert(0, tuple([0] * (len(base) - 1)))
        slices = [
            tuple(
                [
                    slice(i + base, None)  # TODO: if ((i + base >= 0) and (i < in_dim))
                    for i, base, in_dim in zip([0, *offset], base, x.shape)
                ]
            )
            for offset in offsets
        ]

        torch_pad = []
        for pad_value in reversed(base):
            torch_pad = torch_pad + [pad_value, pad_value]
        x_padded = torch.nn.functional.pad(
            x,
            pad=tuple(torch_pad),
            mode=self.padding.get("mode", "constant"),
            value=self.padding.get("value", 0),
        )
        results = (
            tuple([] for _ in range(self.return_tuple)) if self.return_tuple else []
        )
        indexes = []
        for sl in slices:
            patch_set, patch_idx = self._extract_patches(x_padded[sl], self.input_shape)
            patch_set = patch_set.squeeze(1)
            inference = model(patch_set)
            if self.return_tuple:
                for i in range(self.return_tuple):
                    results[i].append(inference[i])
            else:
                results.append(inference)
            indexes.append(patch_idx)
        output_slice = tuple([slice(0, lenght) for lenght in self.ref_shape])
        if self.return_tuple:
            comb_list = []
            for i in range(self.return_tuple):
                comb = self._combine_patches(results[i], offsets, indexes)
                comb = comb[output_slice]
                comb_list.append(comb)
            comb = tuple(comb_list)
        else:
            comb = self._combine_patches(results, offsets, indexes)
            comb = comb[output_slice]
        return comb
