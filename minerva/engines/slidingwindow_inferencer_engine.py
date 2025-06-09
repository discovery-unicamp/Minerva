from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch

from minerva.engines.engine import _Engine, _Inferencer
from minerva.models.nets.base import SimpleSupervisedModel


class SlidingWindowInferencer(_Inferencer):
    def __init__(
        self,
        model: SimpleSupervisedModel,
        strides: Tuple[int, ...],
        input_shape: Tuple[int, ...],
        output_shape: Optional[Tuple[int, ...]] = None,
        weight_function: Optional[Callable[[Tuple[int, ...]], torch.Tensor]] = None,
        padding: Optional[Dict[str, Any]] = None,
        return_tuple: Optional[int] = None,
    ):
        """Wrap a `SimpleSupervisedModel` model's forward method to perform
        inference with sliding windows, transparently traversing the input tensor,
        performing inference in windows, and combining them into a
        single output of the desired size.

        Parameters
        ----------
        model : SimpleSupervisedModel
            Model to be wrapped.
        strides : Tuple[int, ...]
            Sliding window stride for each dimension of the input data, if
            data
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
        self.inferencer = SlidingWindowInferencerEngine(
            strides,
            input_shape,
            output_shape,
            padding,
            weight_function,
            return_tuple,
        )


class SlidingWindowInferencerEngine(_Engine):
    def __init__(
        self,
        strides: Tuple[int, ...],
        input_shape: Tuple[int, ...],
        output_shape: Optional[Tuple[int, ...]] = None,
        padding: Optional[Dict[str, Any]] = None,
        weight_function: Optional[Callable] = None,
        return_tuple: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        strides: Tuple[int, ...]

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
        assert len(input_shape) == len(
            strides
        ), "'input_shape' and 'strides' must have same number of dimensions"
        self.strides = (1, *strides)
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

        if padding is not None:
            assert len(input_shape) == len(
                padding["pad"]
            ), f"Pad tuple does not match expected size ({len(input_shape)})"
            self.padding = padding
            self.padding["pad"] = (0, *self.padding["pad"])
        else:
            self.padding = {"pad": tuple([0] * (len(input_shape) + 1))}
        self.return_tuple = return_tuple

    def _combine_patches(
        self,
        patches: torch.Tensor,
        weight_distribution: torch.Tensor,
        index: Tuple[int, ...],
        ref_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Combines each patch extracted to obtain final result.
        The combination is done by placing each inference result in its respective output place,
        the weight distribution is also placed on a weight accumulator.
        After all patches have been accumulated, the weighted average of the resulting volume is computed.
        """
        accumulator = torch.zeros(ref_shape, device=patches.device)
        weight = (
            torch.zeros(
                tuple([*ref_shape[:1], *ref_shape[2:]]),
                device=patches.device,
            )
            if self.logits_dim
            else torch.zeros(ref_shape, device=patches.device)
        )
        index = tuple([index[0], 1, *index[1:]]) if self.logits_dim else index
        strides = (
            tuple([self.strides[0], 1, *self.strides[1:]])
            if self.logits_dim
            else self.strides
        )
        for patch_index, patch in zip(np.ndindex(index), patches):
            used_len = [
                patch_len + min(0, ref - (idx * stride + patch_len))
                for idx, patch_len, stride, ref in zip(
                    patch_index, self.output_shape, strides, ref_shape
                )
            ]
            sl = [
                slice(idx * stride, idx * stride + patch_len, None)
                for idx, patch_len, stride in zip(patch_index, used_len, strides)
            ]
            used_sl = [slice(0, patch_len, None) for patch_len in used_len[1:]]
            accumulator[tuple(sl)] += patch[tuple(used_sl)]
            if self.logits_dim:
                sl.pop(1)
                used_sl.pop(1)
            weight[tuple(sl)] += weight_distribution[tuple(used_sl)]
        return accumulator / weight

    def _extract_patches(self, data: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int]]:
        """Sliding window patch extraction method."""
        indexes = tuple(
            (
                (np.array(data.shape) - np.array(self.input_shape))
                // np.array(self.strides)
            )
            + 1
        )
        patches = []
        for patch_index in np.ndindex(indexes):
            sl = [
                slice(idx * stride, idx * stride + patch_len, None)
                for idx, patch_len, stride in zip(
                    patch_index, self.input_shape, self.strides
                )
            ]
            patches.append(data[tuple(sl)])
        print("patches: ", len(patches))
        return torch.stack(patches), indexes

    def _compute_output_shape(self, tensor: torch.Tensor) -> Tuple[int]:
        """Computes `SlidingWindowInferencer` output shape based on input tensor shape,
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
        """Perform inference in sliding windows, from the input tensor `x` using the
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

        ref_shape = self._compute_output_shape(x)
        base = self._compute_base_padding(x)

        torch_pad = []
        for pad_value in reversed(base):
            torch_pad = torch_pad + [0, pad_value]
        x_padded = torch.nn.functional.pad(
            x,
            pad=tuple(torch_pad),
            mode=self.padding.get("mode", "constant"),
            value=self.padding.get("value", 0),
        )
        patches, indexes = self._extract_patches(x_padded)
        patches = patches.squeeze(1)
        inference = model(patches)

        if self.weight_function:
            weight_distribution = self.weight_function(self.output_simplified_shape[1:])
            inference = inference * weight_distribution
        else:
            weight_distribution = torch.ones(
                self.output_simplified_shape[1:], device=patches.device
            )

        if self.return_tuple:
            res_list = []

            for i in range(self.return_tuple):
                res = self._combine_patches(
                    inference[i], weight_distribution, indexes, ref_shape
                )
                res_list.append(res)
            res = tuple(res_list)
        else:
            res = self._combine_patches(
                inference, weight_distribution, indexes, ref_shape
            )
        return res
