from .diffusionts_transformer import (
    Transformer,
)
import torch
from typing import Optional
import copy


class DiffusionTSEncoder(Transformer):
    """
    An adapted Transformer from DiffusionTS that functions purely as an encoder. Instead of
    running the full encoder-decoder pipeline, it processes the input up to a certain encoder
    block. For that, it exposes two hyperparameters: timestep and number of encoder blocks.
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
        """
        Initializes the model, reusing the Transformer class.

        New Parameters
        --------------
        diffusion_timestep: int
            The diffusion timestep, by default 0.
        target_block: int, optional
            Number of encoder blocks to apply. If None, defaults to n_layer_enc and
            uses all encoder blocks.
        pass_strategy: str
            The pass strategy to apply, by default 'single'. If 'single' the data
            passes through the model only once, stopping at target_block
            and avoiding the decoder. If 'double', the data passes twice, first
            applying an inferior diffusion_timestep (t-1), and then applying a minimum
            timestep (t=0).
        diffusion_use_t_and_s: bool
            Whether to return the trend and seasonality as features, by default False.
        """
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

    def forward(self, input):
        """Extract encoder features directly or after a denoising pass."""
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
        """Return selected encoder-block features and the input embedding."""
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
        """Reconstruct the signal by combining predicted trend and seasonal residual."""
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
        """Decode embeddings into trend and seasonal residual components."""
        inp_dec = self.pos_dec(emb)
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond)
        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = (
            self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        )
        trend = self.combine_m(mean) + res_m + trend
        return trend, season_error
