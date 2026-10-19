"""Transformer representations from single- and double-pass Diffusion-TS encoders."""

from .diffusionts_transformer import (
    Transformer,
)
import torch
from typing import Optional
import copy


class DiffusionTSEncoder(Transformer):
    """Extract features from a selected number of Diffusion-TS encoder blocks.

    Single pass runs the embedding and encoder blocks at ``diffusion_timestep``.
    Double pass reconstructs a signal with the full encoder-decoder at timestep
    ``diffusion_timestep - 1``, then extracts features at timestep zero using
    independent copies of the embedding, positional encoding, and encoder blocks.

    Parameters
    ----------
    n_feat : int
        Number of input features per timestamp.
    n_channel : int
        Number of timestamps in each input sequence, used by the decoder.
    n_layer_enc : int, optional
        Number of encoder blocks. Default is 5.
    n_layer_dec : int, optional
        Number of decoder blocks used for reconstruction. Default is 14.
    n_embd : int, optional
        Width of the embeddings and encoder features. Default is 1024.
    n_heads : int, optional
        Number of attention heads. Must divide ``n_embd``. Default is 16.
    attn_pdrop : float, optional
        Attention dropout probability. Default is 0.1.
    resid_pdrop : float, optional
        Dropout probability for projections and residual paths. Default is 0.1.
    mlp_hidden_times : int, optional
        Multiplier for the hidden width of each feed-forward block. Default is 4.
    block_activate : str, optional
        Feed-forward activation passed to the Transformer. Default is ``"GELU"``.
    max_len : int, optional
        Positional embedding length; the input must be compatible with this
        positional encoding and the configured sequence length. Default is 2048.
    conv_params : tuple of int, optional
        Kernel size and padding for the seasonal output convolution. If None,
        the Transformer selects them from ``n_feat`` and ``n_channel``.
    diffusion_timestep : int, optional
        Timestep used for single pass. Double pass uses this value minus one
        for reconstruction and zero for feature extraction. Default is 0.
    target_block : int, optional
        Number of encoder blocks to execute, from 1 to ``n_layer_enc`` inclusive.
        If None, runs all encoder blocks. Default is None.
    pass_strategy : {"single", "double"}, optional
        Whether to extract features directly or after reconstruction.
        Default is ``"single"``.
    diffusion_use_t_and_s : bool, optional
        Stored configuration flag. It does not change the returned encoder
        features in this implementation. Default is False.

    Raises
    ------
    ValueError
        If ``pass_strategy`` is unsupported or ``target_block`` is outside
        the encoder block range.

    Notes
    -----
    For double pass, loading a Transformer checkpoint initializes the second
    pass from the loaded first-pass weights. Checkpoints that already contain
    second-pass weights preserve them; strict loading still rejects missing
    or incompatible weights.
    """

    def __init__(
        self,
        n_feat: int,
        n_channel: int,
        n_layer_enc: int = 5,
        n_layer_dec: int = 14,
        n_embd: int = 1024,
        n_heads: int = 16,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_hidden_times: int = 4,
        block_activate: str = "GELU",
        max_len: int = 2048,
        conv_params: Optional[tuple] = None,
        diffusion_timestep: int = 0,
        target_block: Optional[int] = None,
        pass_strategy: str = "single",
        diffusion_use_t_and_s: bool = False,
    ):
        """Build the Transformer and configure the feature extraction passes."""
        if pass_strategy not in ("single", "double"):
            raise ValueError("pass_strategy must be 'single' or 'double'.")

        super().__init__(
            n_feat=n_feat,
            n_channel=n_channel,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            n_embd=n_embd,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            block_activate=block_activate,
            max_len=max_len,
            conv_params=conv_params,
        )

        self.timestep = diffusion_timestep
        self.pass_strategy = pass_strategy
        self.diffusion_use_t_and_s = diffusion_use_t_and_s
        self.n_layer_enc = n_layer_enc
        if target_block is None:
            target_block = n_layer_enc

        if target_block <= 0 or target_block > n_layer_enc:
            raise ValueError("Incorrect value for target_block")
        self.diffusion_encoder_blocks = target_block

        self.additional_emb = None
        self.additional_pos_enc = None
        self.additional_encoder_blocks = None
        if pass_strategy == "double":
            self.additional_emb = copy.deepcopy(self.emb)
            self.additional_pos_enc = copy.deepcopy(self.pos_enc)
            self.additional_encoder_blocks = copy.deepcopy(self.encoder.blocks)
            self.register_load_state_dict_pre_hook(self._initialize_second_pass_weights)

    def _initialize_second_pass_weights(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Initialize second-pass weights when loading a Transformer checkpoint.

        Existing second-pass keys identify an encoder checkpoint and are left
        untouched, including incomplete checkpoints that strict loading rejects.
        The prefix supports loading the encoder inside a supervised model.
        """
        copies = {
            "emb.": "additional_emb.",
            "pos_enc.": "additional_pos_enc.",
            "encoder.blocks.": "additional_encoder_blocks.",
        }
        additional_prefixes = tuple(prefix + name for name in copies.values())
        if any(key.startswith(additional_prefixes) for key in state_dict):
            return

        for source, destination in copies.items():
            source_prefix = prefix + source
            for key, value in list(state_dict.items()):
                if key.startswith(source_prefix):
                    new_key = prefix + destination + key[len(source_prefix) :]
                    state_dict[new_key] = value.clone()

    def forward(self, input):
        """Extract encoder features with the configured pass strategy.

        Parameters
        ----------
        input : torch.Tensor
            Signal batch of shape ``(batch_size, sequence_length, n_feat)``.

        Returns
        -------
        torch.Tensor
            Encoder features of shape ``(batch_size, sequence_length, n_embd)``.
            The temporal dimension is retained without flattening or pooling.
        """
        t = torch.full((len(input),), self.timestep, device=input.device).long()
        if self.pass_strategy == "single":
            # Single pass
            embedding, _ = self.encoder_partial_forward(
                input, t, self.diffusion_encoder_blocks, additional=False
            )
        elif self.pass_strategy == "double":
            # First pass
            z = self.model_pass_forward(input, t - 1)
            t_0 = torch.full((len(input),), 0, device=input.device).long()
            # Second pass
            embedding, _ = self.encoder_partial_forward(
                z, t_0, self.diffusion_encoder_blocks, additional=True
            )
        return embedding

    def encoder_partial_forward(self, input, t, encoder_block, additional=False):
        # Which encoder to use: original (single pass) or additional (double pass)
        """Apply the input embedding and a selected number of encoder blocks.

        Parameters
        ----------
        input : torch.Tensor
            Signals of shape ``(batch_size, sequence_length, n_feat)``.
        t : torch.Tensor
            Diffusion timesteps of shape ``(batch_size,)``.
        encoder_block : int
            Number of encoder blocks to execute.
        additional : bool, optional
            Use the second-pass modules created for ``pass_strategy="double"``.
            Default is False.

        Returns
        -------
        torch.Tensor
            Encoder features of shape ``(batch_size, sequence_length, n_embd)``.
        torch.Tensor
            Input projection before positional encoding, with the same shape.
        """
        emb_module = self.emb
        pos_enc_module = self.pos_enc
        encoder_blocks_module = self.encoder.blocks
        if additional:
            emb_module = self.additional_emb
            pos_enc_module = self.additional_pos_enc
            encoder_blocks_module = self.additional_encoder_blocks
        # Partial forward pass through the encoder
        input_emb = emb_module(input)
        inp_enc = pos_enc_module(input_emb)
        for block_idx in range(encoder_block):
            inp_enc, _ = encoder_blocks_module[block_idx](inp_enc, t)
        return inp_enc, input_emb

    def model_pass_forward(self, input, t):
        """Reconstruct a signal with the full first-pass encoder and decoder.

        Parameters
        ----------
        input : torch.Tensor
            Signals of shape ``(batch_size, sequence_length, n_feat)``.
        t : torch.Tensor
            Diffusion timesteps of shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor
            Reconstructed signal of shape ``(batch_size, sequence_length, n_feat)``,
            obtained by combining trend and seasonal residual components.
        """
        encoder_output, input_emb = self.encoder_partial_forward(
            input, t, self.n_layer_enc, additional=False
        )
        inp_dec = self.pos_dec(input_emb)
        output, mean, trend, season = self.decoder(inp_dec, t, encoder_output)
        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = (
            self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        )
        trend = self.combine_m(mean) + res_m + trend
        return trend + season_error

    def decoder_forward(self, emb, enc_cond, t):
        """Decode input embeddings into trend and seasonal residual components.

        Parameters
        ----------
        emb : torch.Tensor
            Input embeddings before positional encoding, of shape
            ``(batch_size, sequence_length, n_embd)``.
        enc_cond : torch.Tensor
            Encoder conditioning features with the same shape as ``emb``.
        t : torch.Tensor
            Diffusion timesteps of shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor
            Trend component of shape ``(batch_size, sequence_length, n_feat)``.
        torch.Tensor
            Seasonal residual component with the same shape as the trend.
        """
        inp_dec = self.pos_dec(emb)
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond)
        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = (
            self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        )
        trend = self.combine_m(mean) + res_m + trend
        return trend, season_error
