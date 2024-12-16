from typing import List, Tuple, Optional, Dict, Any
import torch
import numpy as np
import lightning as L


class BasePatchInferencer(L.LightningModule):
    """Inference in patches for models

    This class provides utility methods for performing inference in patches
    """

    def __init__(
        self,
        model: L.LightningModule,
        input_shape: Tuple,
        output_shape: Optional[Tuple] = None,
        weight_function: Optional[callable] = None,
        offsets: Optional[List[Tuple]] = None,
        padding: Optional[Dict[str, Any]] = None,
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
                mode (optional): 'constant', 'reflect', 'replicate' or 'cicular'. Defaults to 'constant'.
                value (optional): fill value for 'constante'. Defaults to 0.
        """
        super().__init__()
        self.model = model.eval()
        self.input_shape = input_shape
        self.output_shape = (
            output_shape if output_shape is not None else input_shape
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
        else:
            self.padding = {"pad": tuple([0] * len(input_shape))}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def _reconstruct_patches(
        self,
        patches: torch.Tensor,
        index: Tuple[int],
        weights: bool,
        inner_dim: int = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Rearranges patches to reconstruct area of interest from patches and weights
        """
        reconstruct_shape = np.array(self.output_shape) * np.array(index)
        if weights:
            weight = torch.zeros(tuple(reconstruct_shape))
            base_weight = (
                self.weight_function(self.input_shape)
                if self.weight_function
                else torch.ones(self.input_shape)
            )
        else:
            weight = None
        if inner_dim is not None:
            reconstruct_shape = np.append(reconstruct_shape, inner_dim)
        reconstruct = torch.zeros(tuple(reconstruct_shape))
        for patch_index, patch in zip(np.ndindex(index), patches):
            sl = [
                slice(idx * patch_len, (idx + 1) * patch_len, None)
                for idx, patch_len in zip(patch_index, self.input_shape)
            ]
            if weights:
                weight[tuple(sl)] = base_weight
            if inner_dim is not None:
                sl.append(slice(None, None, None))
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
        has_inner_dim = len(offset) < len(arrays[0].shape)
        pad_width = []
        sl = []
        ref_shape = list(ref_shape)
        arr_shape = list(arrays[0].shape)
        if has_inner_dim:
            arr_shape = arr_shape[:-1]
        for idx, lenght, ref in zip(offset, arr_shape, ref_shape):
            if idx > 0:
                sl.append(slice(0, min(lenght, ref), None))
                pad_width = [idx, max(ref - lenght - idx, 0)] + pad_width
            else:
                sl.append(slice(np.abs(idx), min(lenght, ref - idx), None))
                pad_width = [0, max(ref - lenght - idx, 0)] + pad_width
        adjusted = [
            (
                torch.nn.functional.pad(
                    arr[tuple([*sl, slice(None, None, None)])],
                    pad=tuple([0, 0, *pad_width]),
                    mode="constant",
                    value=pad_value,
                )
                if has_inner_dim
                else torch.nn.functional.pad(
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
        How results are combined is dependent on what is being combined.
        RegressionPatchInferencer uses Weighted Average
        ClassificationPatchInferencer uses Voting (hard or soft)
        """
        raise NotImplementedError("Combine patches method must be implemented")

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform Inference in Patches

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor.
        """
        assert len(x.shape) == len(
            self.input_shape
        ), "Input and self.input_shape sizes must match"

        self.ref_shape = self._compute_output_shape(x)
        offsets = list(self.offsets)
        base = self.padding["pad"]
        offsets.insert(0, tuple([0] * len(base)))

        slices = [
            tuple(
                [
                    slice(
                        i + base, None
                    )  # TODO: if ((i + base >= 0) and (i < in_dim))
                    for i, base, in_dim in zip(offset, base, x.shape)
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
        results = []
        indexes = []
        for sl in slices:
            patch_set, patch_idx = self._extract_patches(
                x_padded[sl], self.input_shape
            )
            r = self.model(patch_set)
            if len(r.shape) == 4:
                r = r.unsqueeze(1)
                r = r.permute(0, 1, 3, 4, 2).contiguous()
            results.append(r)
            indexes.append(patch_idx)
        output_slice = tuple([slice(0, lenght) for lenght in x.shape])
        return self._combine_patches(results, offsets, indexes)[output_slice]

    def predict_step(self, batch, *args, **kwargs):
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = batch[0]
        return torch.stack([self(x) for x in batch])

class WeightedAvgPatchInferencer(BasePatchInferencer):
    """
    PatchInferencer with Weighted Average combination function.
    """

    def _combine_patches(
        self,
        results: List[torch.Tensor],
        offsets: List[Tuple[int]],
        indexes: List[Tuple[int]],
    ) -> torch.Tensor:
        reconstructed = []
        weights = []
        for patches, offset, shape in zip(results, offsets, indexes):
            reconstruct, weight = self._reconstruct_patches(
                patches, shape, weights=True
            )
            reconstruct, weight = self._adjust_patches(
                [reconstruct, weight], self.ref_shape, offset
            )

            reconstructed.append(reconstruct)
            weights.append(weight)
        reconstructed = torch.stack(reconstructed, dim=0)
        weights = torch.stack(weights, dim=0)
        return torch.sum(reconstructed * weights, dim=0) / torch.sum(
            weights, dim=0
        )


class VotingPatchInferencer(BasePatchInferencer):
    """
    PatchInferencer with Voting combination function.
    Note: Models used with VotingPatchInferencer must return class probabilities in inner dimension
    """

    def __init__(
        self,
        model: L.LightningModule,
        num_classes: int,
        input_shape: Tuple,
        output_shape: Optional[Tuple] = None,
        weight_function: Optional[callable] = None,
        offsets: Optional[List[Tuple]] = None,
        padding: Optional[Dict[str, Any]] = None,
        voting: str = "soft",
    ):
        """Initialize the patch inference auxiliary class

        Parameters
        ----------
        model : L.LightningModule
            Model used in inference.
        num_classes: int
            number of classes of the classification task
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
                mode (optional): 'constant', 'reflect', 'replicate' or 'cicular'. Defaults to 'constant'.
                value (optional): fill value for 'constante'. Defaults to 0.
        voting: str
            voting method to use, can be either 'soft'or 'hard'. Defaults to 'soft'.
        """
        super().__init__(
            model, input_shape, output_shape, weight_function, offsets, padding
        )
        assert voting in [
            "soft",
            "hard",
        ], "voting should be either 'soft' or 'hard'"
        self.num_classes = num_classes
        self.voting = voting

    def _combine_patches(
        self,
        results: List[torch.Tensor],
        offsets: List[Tuple[int]],
        indexes: List[Tuple[int]],
    ) -> torch.Tensor:
        voting_method = getattr(self, f"_{self.voting}_voting")
        return voting_method(results, offsets, indexes)

    def _hard_voting(
        self,
        results: List[torch.Tensor],
        offsets: List[Tuple[int]],
        indexes: List[Tuple[int]],
    ) -> torch.Tensor:
        """
        Hard voting combination function
        """
        # torch.mode does not work like scipy.stats.mode
        raise NotImplementedError("Hard voting not yet supported")
        # reconstructed = []
        # for patches, offset, shape in zip(results, offsets, indexes):
        #     reconstruct, _ = self._reconstruct_patches(
        #         patches, shape, weights=False, inner_dim=self.num_classes
        #     )
        #     reconstruct = torch.argmax(reconstruct, dim=-1).float()
        #     reconstruct = self._adjust_patches(
        #         [reconstruct], self.ref_shape, offset, pad_value=torch.nan
        #     )[0]
        #     reconstructed.append(reconstruct)
        # reconstructed = torch.stack(reconstructed, dim=0)
        # ret = torch.mode(reconstructed, dim=0, keepdims=False)[
        #     0
        # ]  # TODO check behaviour on GPU, according to issues may have nonsense results
        # return ret

    def _soft_voting(
        self,
        results: List[torch.Tensor],
        offsets: List[Tuple[int]],
        indexes: List[Tuple[int]],
    ) -> torch.Tensor:
        """
        Soft voting combination function
        """
        reconstructed = []
        for patches, offset, shape in zip(results, offsets, indexes):
            # print(f"Patches shape: {patches.shape}, Offset: {offset}, Shape: {shape}")
            reconstruct, _ = self._reconstruct_patches(
                patches, shape, weights=False, inner_dim=self.num_classes
            )
            reconstruct = self._adjust_patches(
                [reconstruct], self.ref_shape, offset
            )[0]
            reconstructed.append(reconstruct)
        reconstructed = torch.stack(reconstructed, dim=0)
        return torch.argmax(torch.sum(reconstructed, dim=0), dim=-1)
