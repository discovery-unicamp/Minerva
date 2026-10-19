"""TS2Vec contrastive representation learning and time-series preprocessing utilities.

Arrays use the layout ``(instances, timestamps, features)`` and represent
missing values with NaN. The model exposes its own training and encoding
methods rather than the Lightning training interface.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from minerva.models.nets.tnc import TSEncoder
import math


def instance_contrastive_loss(z1, z2):
    """Contrast matching instances across two views at each timestamp.

    Parameters
    ----------
    z1 : torch.Tensor
        First-view representations of shape ``(batch_size, timestamps, channels)``.
    z2 : torch.Tensor
        Second-view representations with the same shape as ``z1``.

    Returns
    -------
    torch.Tensor
        Scalar contrastive loss averaged over matching pairs and timestamps.
        Returns zero when the batch contains only one instance.
    """
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    """Contrast matching timestamps across two views within each instance.

    Parameters
    ----------
    z1 : torch.Tensor
        First-view representations of shape ``(batch_size, timestamps, channels)``.
    z2 : torch.Tensor
        Second-view representations with the same shape as ``z1``.

    Returns
    -------
    torch.Tensor
        Scalar contrastive loss averaged over instances and matching timestamps.
        Returns zero when the sequence contains only one timestamp.
    """
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    """Combine instance and temporal contrastive losses at multiple time scales.

    Successive max-pooling operations halve the temporal dimension. The total
    loss is averaged over the evaluated scales, including the final single
    timestamp when present.

    Parameters
    ----------
    z1 : torch.Tensor
        First-view representations of shape ``(batch_size, timestamps, channels)``.
    z2 : torch.Tensor
        Second-view representations with the same shape as ``z1``.
    alpha : float, optional
        Weight of the instance loss. The temporal loss has weight ``1 - alpha``.
        Default is 0.5.
    temporal_unit : int, optional
        First pooling level at which to include temporal contrast. Level zero
        uses the original temporal resolution. Default is 0.

    Returns
    -------
    torch.Tensor
        Scalar loss averaged over the temporal hierarchy.
    """
    loss = torch.tensor(0.0, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def torch_pad_nan(arr, left=0, right=0, dim=0):
    """Pad a tensor with NaN values along a selected dimension.

    Parameters
    ----------
    arr : torch.Tensor
        Input tensor. Padding tensors are created on the CPU.
    left : int, optional
        Number of entries to prepend. Default is 0.
    right : int, optional
        Number of entries to append. Default is 0.
    dim : int, optional
        Dimension along which to pad. Default is 0.

    Returns
    -------
    torch.Tensor
        Tensor with the requested padding. If both padding sizes are
        nonpositive, returns the input unchanged.
    """
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def centerize_vary_length_series(x):
    """Center each series according to its leading and trailing missing timestamps.

    Parameters
    ----------
    x : numpy.ndarray
        Series of shape ``(instances, timestamps, features)``. A timestamp is
        considered missing when all its features are NaN.

    Returns
    -------
    numpy.ndarray
        Series with the same shape, circularly shifted along time so that
        leading and trailing missing regions are balanced.
    """
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[: x.shape[0], : x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    """Pad a floating-point array with NaN values to a target length.

    Parameters
    ----------
    array : numpy.ndarray
        Input array with dtype float16, float32, or float64.
    target_length : int
        Minimum length of the selected axis. Longer arrays are not truncated.
    axis : int, optional
        Axis to pad. Default is 0.
    both_side : bool, optional
        Split padding between both ends. If False, append all padding.
        Default is False.

    Returns
    -------
    numpy.ndarray
        Padded array, or the input unchanged when no padding is needed.

    Raises
    ------
    AssertionError
        If the array does not have a supported floating-point dtype.
    """
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode="constant", constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    """Split an array into sections and pad shorter sections with NaN values.

    Parameters
    ----------
    x : numpy.ndarray
        Input array with dtype float16, float32, or float64.
    sections : int
        Number of sections passed to ``numpy.array_split``.
    axis : int, optional
        Axis along which to split and pad. Default is 0.

    Returns
    -------
    list of numpy.ndarray
        Sections padded at the end to match the first section's length.

    Raises
    ------
    AssertionError
        If the array does not have a supported floating-point dtype.
    """
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def take_per_row(A, indx, num_elem):
    """Extract a contiguous time window with an individual offset for each instance.

    Parameters
    ----------
    A : torch.Tensor
        Input batch of shape ``(batch_size, timestamps, features)``.
    indx : numpy.ndarray
        Starting indices of shape ``(batch_size,)``. Each index must leave
        enough entries for the requested window.
    num_elem : int
        Number of timestamps to extract per instance.

    Returns
    -------
    torch.Tensor
        Windows of shape ``(batch_size, num_elem, features)``.
    """
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


class TS2Vec:
    """Learn time-series representations with hierarchical contrastive learning.

    Training compares overlapping crops at multiple temporal resolutions.
    Inference uses an averaged copy of the encoder, with optional temporal
    pooling and sliding windows.

    Parameters
    ----------
    input_dims : int
        Number of input features per timestamp. Use 1 for univariate series.
    output_dims : int, optional
        Number of representation features per timestamp. Default is 320.
    hidden_dims : int, optional
        Hidden width of the convolutional encoder. Default is 64.
    depth : int, optional
        Encoder depth passed to ``TSEncoder``. Default is 10.
    device : str or torch.device, optional
        Device for training and inference. Default is ``"cuda"``.
    lr : float, optional
        AdamW learning rate. Default is 0.001.
    batch_size : int, optional
        Training batch size and default encoding batch size. Default is 16.
    max_train_length : int, optional
        Maximum crop length used during training. Long series may first be
        split into sections; remaining long batches are cropped randomly.
        If None, no length limit is applied. Default is None.
    temporal_unit : int, optional
        First pooling level at which to include temporal contrast. The input
        crop must contain at least ``2 ** (temporal_unit + 1)`` timestamps.
        Default is 0.
    after_iter_callback : callable, optional
        Callback invoked as ``callback(model, loss)`` after each optimizer
        update, with the scalar loss as a float. Default is None.
    after_epoch_callback : callable, optional
        Callback invoked as ``callback(model, mean_loss)`` after each completed
        epoch. Default is None.

    Attributes
    ----------
    net : torch.optim.swa_utils.AveragedModel
        Averaged encoder used by ``encode``, ``save``, and ``load``.
    n_epochs : int
        Number of completed training epochs across calls to ``fit``.
    n_iters : int
        Number of optimizer updates across calls to ``fit``.
    """

    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device="cuda",
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
    ):
        """Initialize the trainable encoder, averaged encoder, and training counters."""

        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self._net = TSEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
        ).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        """Train the encoder on pairs of overlapping time-series crops.

        Parameters
        ----------
        train_data : numpy.ndarray
            Floating-point data of shape ``(instances, timestamps, input_dims)``.
            Missing values must be NaN. Entirely missing instances are discarded.
        n_epochs : int, optional
            Total completed-epoch limit, including previous calls to ``fit``.
            If both limits are provided, training stops when either is reached.
            Default is None.
        n_iters : int, optional
            Total optimizer-update limit, including previous calls to ``fit``.
            If both limits are None, uses 200 for arrays with at most 100000
            elements, or 600 otherwise. Default is None.
        verbose : bool, optional
            Print the mean loss after each completed epoch. Default is False.

        Returns
        -------
        list of float
            Mean losses for epochs completed during this call. An epoch interrupted
            by the iteration limit is not included.

        Raises
        ------
        AssertionError
            If ``train_data`` is not three-dimensional.

        Notes
        -----
        Each call creates a new AdamW optimizer. Encoder weights, averaged weights,
        and training counters are retained between calls, but optimizer state is not.
        """
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = (
                200 if train_data.size <= 100000 else 600
            )  # default param for n_iters

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(
                    split_with_nan(train_data, sections, axis=1), axis=0
                )

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)

        loss_log = []

        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = batch[0]
                if (
                    self.max_train_length is not None
                    and x.size(1) > self.max_train_length
                ):
                    window_offset = np.random.randint(
                        x.size(1) - self.max_train_length + 1
                    )
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                ts_l = x.size(1)
                crop_l = np.random.randint(
                    low=2 ** (self.temporal_unit + 1), high=ts_l + 1
                )
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(
                    low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0)
                )

                optimizer.zero_grad()

                out1 = self._net(
                    take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
                )
                out1 = out1[:, -crop_l:]

                out2 = self._net(
                    take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
                )
                out2 = out2[:, :crop_l]

                loss = hierarchical_contrastive_loss(
                    out1, out2, temporal_unit=self.temporal_unit
                )

                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                n_epoch_iters += 1

                self.n_iters += 1

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        return loss_log

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        """Encode a batch and optionally pool or slice its temporal representations.

        Parameters
        ----------
        x : torch.Tensor
            Series of shape ``(batch_size, timestamps, input_dims)``.
        mask : str, optional
            Masking option forwarded to ``TSEncoder``. Default is None.
        slicing : slice, optional
            Temporal region to retain. For full-series pooling, slicing is applied
            before pooling; otherwise it is applied afterwards. Default is None.
        encoding_window : str or int, optional
            Use ``"full_series"`` for global max pooling, an integer for local
            max pooling, or ``"multiscale"`` to concatenate pooled features from
            several window sizes. None returns unpooled features. Default is None.

        Returns
        -------
        torch.Tensor
            CPU representations retaining a temporal axis. Full-series pooling
            returns shape ``(batch_size, 1, output_dims)``; multiscale pooling
            concatenates scales along the feature axis.
        """
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == "full_series":
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2,
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == "multiscale":
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p,
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(
        self,
        data,
        mask=None,
        encoding_window=None,
        causal=False,
        sliding_length=None,
        sliding_padding=0,
        batch_size=None,
    ):
        """Compute representations with the averaged encoder in evaluation mode.

        Parameters
        ----------
        data : numpy.ndarray
            Series of shape ``(instances, timestamps, input_dims)``, with NaN
            indicating missing values.
        mask : str, optional
            Masking option forwarded to ``TSEncoder``. Its current implementation
            explicitly applies random masking for ``"binomial"``. Default is None.
        encoding_window : str or int, optional
            Temporal pooling option: ``"full_series"``, ``"multiscale"``, or an
            integer window size. None retains timestamp-level representations.
            Default is None.
        causal : bool, optional
            Omit right-hand context padding when sliding windows are used.
            This option does not change the encoder architecture. Default is False.
        sliding_length : int, optional
            Number of output timestamps per sliding window. If None, encode each
            batch without sliding windows. Default is None.
        sliding_padding : int, optional
            Number of context timestamps on each side of a sliding window; causal
            mode omits the right-hand context. Default is 0.
        batch_size : int, optional
            Encoding batch size. If None, use the configured training batch size.
            Default is None.

        Returns
        -------
        numpy.ndarray
            Representations with shape determined by pooling and sliding options.
            Without sliding, no pooling returns ``(instances, timestamps,
            output_dims)``, and full-series pooling returns ``(instances,
            output_dims)``. Multiscale pooling concatenates features across scales.

        Raises
        ------
        AssertionError
            If the averaged encoder is unavailable or ``data`` is not
            three-dimensional.

        Notes
        -----
        Sliding windows are NaN-padded at sequence boundaries. The final window
        may retain padded timestamps when the length is not a multiple of
        ``sliding_length``. The encoder's previous training mode is restored
        when encoding completes successfully.
        """
        assert self.net is not None, "please train or load a net first"
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1,
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(
                                        sliding_padding,
                                        sliding_padding + sliding_length,
                                    ),
                                    encoding_window=encoding_window,
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(
                                    sliding_padding, sliding_padding + sliding_length
                                ),
                                encoding_window=encoding_window,
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(
                                    sliding_padding, sliding_padding + sliding_length
                                ),
                                encoding_window=encoding_window,
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == "full_series":
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(
                        x, mask, encoding_window=encoding_window
                    )
                    if encoding_window == "full_series":
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()

    def save(self, fn):
        """Save the averaged encoder weights to a file.

        Parameters
        ----------
        fn : str or os.PathLike
            Destination accepted by ``torch.save``.

        Notes
        -----
        The file contains the averaged encoder state only. It does not contain
        optimizer state, training counters, or model constructor settings.
        """
        torch.save(self.net.state_dict(), fn)

    def load(self, fn):
        """Load averaged encoder weights from a compatible saved state dictionary.

        Parameters
        ----------
        fn : str or os.PathLike
            Source accepted by ``torch.load``. Tensors are mapped to ``self.device``.

        Notes
        -----
        Construct the model with matching encoder dimensions before loading.
        This method updates the averaged encoder used for inference; it does not
        restore the trainable encoder, optimizer, or training counters.
        """
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
