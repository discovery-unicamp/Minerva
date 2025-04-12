from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch

from minerva.engines.engine import _Engine
from minerva.models.nets.base import SimpleSupervisedModel


class PatchInferencer(L.LightningModule):
    """This class acts as a normal `L.LightningModule` that wraps a
    `SimpleSupervisedModel` model allowing it to perform inference in patches.
    This is useful when the model's default input size is smaller than the
    desired input size (sample size). In this case, the engine split the input
    tensor into patches, perform inference in each patch, and combine them into
    a single output of the desired size. The combination of patches can be
    parametrized by a `weight_function` allowing a customizable combination of
    patches (e.g, combining using weighted average). It is important to note
    that only model's forward are wrapped, and, thus, any method that requires
    the forward method (e.g., training_step, predict_step) will be performed in
    patches, transparently to the user.
    """

    def __init__(
        self,
        model: SimpleSupervisedModel,
        input_shape: Tuple[int, ...],
        output_shape: Optional[Tuple[int, ...]] = None,
        weight_function: Optional[Callable[[Tuple[int, ...]], torch.Tensor]] = None,
        offsets: Optional[List[Tuple[int, ...]]] = None,
        padding: Optional[Dict[str, Any]] = None,
        return_tuple: Optional[int] = None,
    ):
        """Wrap a `SimpleSupervisedModel` model's forward method to perform
        inference in patches, transparently splitting the input tensor into
        patches, performing inference in each patch, and combining them into a
        single output of the desired size.

        Parameters
        ----------
        model : SimpleSupervisedModel
            Model to be wrapped.
        input_shape : Tuple[int, ...]
            Expected input shape of the wrapped model.
        output_shape : Tuple[int, ...], optional
            Expected output shape of the wrapped model. For models that return
            logits (e.g., classification models), the `output_shape` must
            include an  additional dimension at the beginning to accommodate
            the number of output classes. For example, if the model processes
            an input tensor of shape (1, 128, 128) and outputs logits for 10
            classes, the expected `output_shape` should be (10, 1, 128, 128).
            If the model does not return logits (e.g., return a tensor after
            applying an `argmax` operation, or a regression models that usually
            returns a tensor with the same shape as the input tensor), the
            `output_shape` should have the same number of dimensions as the
            input shape. Defaults to None, which assumes the output shape is
            the same as the `input_shape` parameter.
        weight_function: Callable[[Tuple[int, ...]], torch.Tensor], optional
            Function that receives a tensor shape and returns the weights for
            each position of a tensor with the given shape. Useful when regions
            of the inference present diminishing performance when getting
            closer to borders, for instance.
        offsets : List[Tuple[int, ...]], optional
            List of tuples with offsets that determine the shift of the initial
            position of the patch subdivision.
        padding : Dict[str, Any], optional
            Dictionary describing padding strategy. Keys:
                - pad (mandatory): tuple with pad width (int) for each
                    dimension, e.g.(0, 3, 3) when working with a tensor with 3
                    dimensions.
                - mode (optional): 'constant', 'reflect', 'replicate' or
                    'circular'. Defaults to 'constant'.
                - value (optional): fill value for 'constant'. Defaults to 0.
            If None, no padding is applied.
        return_tuple: int, optional
            Some models may return multiple outputs for a single sample (e.g.,
            outputs from multiple auxiliary heads). This parameter is a integer
            that defines the number of outputs the model generates. By default,
            it is None, which indicates that the model produces a single output
            for a single input. When set, it indicates the number of outputs
            the model produces.
        """
        super().__init__()
        self.model = model
        self.patch_inferencer = PatchInferencerEngine(
            input_shape,
            output_shape,
            offsets,
            padding,
            weight_function,
            return_tuple,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform inference in patches.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input data.
        """
        return self.patch_inferencer(self.model, x)

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:
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


class PatchInferencerEngine(_Engine):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Optional[Tuple[int, ...]] = None,
        offsets: Optional[List[Tuple[int, ...]]] = None,
        padding: Optional[Dict[str, Any]] = None,
        weight_function: Optional[Callable] = None,
        return_tuple: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of each patch to process.
        output_shape : Tuple[int, ...], optional
            Expected output shape of the model. For models that return logits,
            the `output_shape` must include an additional dimension at the
            beginning to accommodate the number of output classes. Else, the
            `output_shape` should have the same number of dimensions as the
            `input_shape` (i.e., no logits are returned). Defaults to
            input_shape.
        padding : Dict[str, Any], optional
            Padding configuration with keys:
                - 'pad': Tuple of padding for each expected final dimension,
                    e.g., (0, 512, 512) - (c, h, w).
                - 'mode': Padding mode, e.g., 'constant', 'reflect'.
                - 'value': Padding value if mode is 'constant'.
            Defaults to None, which means no padding is applyied.
        weight_function : Callable, optional
            Function to calculate the weight of each patch. Defaults to None.
        return_tuple : int, optional
            Number of outputs to return. This is useful when the model returns
            multiple outputs for a single input (e.g., from multiple auxiliary
            heads). Defaults to None.
        """
        self.input_shape = (1, *input_shape)
        self.output_shape = (
            (1, *output_shape) if output_shape is not None else self.input_shape
        )

        # Check if possible classification task (has logits)
        self.logits_dim = len(self.input_shape) != len(self.output_shape)
        self.output_simplified_shape = (
            tuple([*self.output_shape[:1], *self.output_shape[2:]])
            if self.logits_dim
            else self.output_shape
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
        """Rearranges patches to reconstruct area of interest from patches and
        weights.
        """
        index = tuple([index[0], 1, *index[1:]]) if self.logits_dim else index
        reconstruct_shape = np.array(self.output_shape) * np.array(index)

        weight = (
            torch.zeros(
                tuple([*reconstruct_shape[:1], *reconstruct_shape[2:]]),
                device=patches.device,
            )
            if self.logits_dim
            else torch.zeros(tuple(reconstruct_shape), device=patches.device)
        )

        base_weight = (
            self.weight_function(self.output_simplified_shape)
            if self.weight_function
            else torch.ones(self.output_simplified_shape, device=patches.device)
        )

        reconstruct = torch.zeros(tuple(reconstruct_shape), device=patches.device)
        for patch_index, patch in zip(np.ndindex(index), patches):
            sl = [
                slice(idx * patch_len, (idx + 1) * patch_len, None)
                for idx, patch_len in zip(patch_index, self.output_shape)
            ]
            reconstruct[tuple(sl)] = patch
            if self.logits_dim:
                sl.pop(1)
            weight[tuple(sl)] = base_weight
        if self.logits_dim:
            weight = weight.unsqueeze(1)
        return reconstruct, weight

    def _adjust_patches(
        self,
        arrays: List[torch.Tensor],
        ref_shape: Tuple[int],
        offset: Tuple[int],
        pad_value: int = 0,
    ) -> List[torch.Tensor]:
        """Pads reconstructed patches with `pad_value` to have same shape as
        the reference shape from the base patch set.
        """
        pad_width = []
        sl = []
        ref_shape = list(ref_shape)
        arr_shape = list(arrays[0].shape)
        adjusted_offset = [0, 0, *offset] if self.logits_dim else [0, *offset]
        for idx, length, ref in zip(adjusted_offset, arr_shape, ref_shape):
            if idx > 0:
                sl.append(slice(0, min(length, ref - idx), None))
                pad_width = [idx, max(ref - length - idx, 0)] + pad_width
            else:
                sl.append(slice(np.abs(idx), min(length, ref - idx), None))
                pad_width = [0, max(ref - length - idx, 0)] + pad_width
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
        """Performs the combination of patches based on the weight function."""
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
        """Patch extraction method. It will be called once for the base patch
        set and also for the requested offsets (overlapping patch sets).
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
        """Computes `PatchInferencer` output shape based on input tensor shape,
        and model's input and output shapes.
        """
        if self.input_shape == self.output_shape:
            return tensor.shape
        shape = []
        for i, o, t in zip(
            self.input_shape, self.output_simplified_shape, tensor.shape
        ):
            if i != o:
                shape.append(int(t * o // i))
            else:
                shape.append(t)

        if self.logits_dim:
            shape.insert(1, self.output_shape[1])

        return tuple(shape)

    def _compute_base_padding(self, tensor: torch.Tensor):
        """Computes the padding for the base patch set based on the input
        tensor shape and the model's input shape.
        """
        padding = [0, 0]
        for i, t in zip(self.padding["pad"][2:], tensor.shape[2:]):
            padding.append(max(0, i - t))
        return padding

    def __call__(
        self, model: Union[L.LightningModule, torch.nn.Module], x: torch.Tensor
    ):
        """Perform inference in patches, from the input tensor `x` using the
        model `model`.

        Parameters
        ----------
        model: Union[L.LightningModule, torch.nn.Module]
            Model to perform inference.
        x : torch.Tensor
            Input tensor of the sample. It can be a single sample or a batch
            of samples.
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
                    slice(i, None)  # TODO: if ((i + base >= 0) and (i < in_dim))
                    for i, in_dim in zip([0, *offset], x.shape)
                ]
            )
            for offset in offsets
        ]

        torch_pad = []
        for pad_value in reversed(base):
            torch_pad = torch_pad + [0, pad_value]
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
        output_slice = tuple([slice(0, length) for length in self.ref_shape])
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
