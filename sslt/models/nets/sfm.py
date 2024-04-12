import torch
import torch.nn as nn
import pytorch_lightning as pl
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block
import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model,newsize1=None,newsize2=None):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            if newsize1 == None:
                newsize1,newsize2 = new_size,new_size
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, newsize1, newsize2))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(newsize1, newsize2), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
        # elif orig_size > new_size:
        #     print("Position generate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        #     pos_tokens = get_2d_sincos_pos_embed(embedding_size, new_size, cls_token=True)
        #     pos_tokens = torch.from_numpy(pos_tokens).float().unsqueeze(0)
        #     checkpoint_model['pos_embed'] = pos_tokens




class MaskedAutoencoderViT(pl.LightningModule):
    """
    Masked Autoencoder with VisionTransformer backbone.

    Args:
        img_size (int): Size of input image.
        patch_size (int): Size of image patch.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of token embeddings.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        decoder_embed_dim (int): Dimension of decoder embeddings.
        decoder_depth (int): Number of decoder transformer blocks.
        decoder_num_heads (int): Number of decoder attention heads.
        mlp_ratio (float): Ratio of MLP hidden layer size to embedding size.
        norm_layer (torch.nn.LayerNorm): Normalization layer.
        norm_pix_loss (bool): Whether to normalize pixel loss.

    References:
        - timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
        - DeiT: https://github.com/facebookresearch/deit
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.in_chans = in_chans
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # Initialization
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Extract patches from input images.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).

        Returns:
            torch.Tensor: Patches of shape (N, num_patches, patch_size^2 * in_chans).
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape((imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape((imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        Reconstruct images from patches.

        Args:
            x (torch.Tensor): Patches of shape (N, L, patch_size^2 * in_chans).

        Returns:
            torch.Tensor: Reconstructed images of shape (N, C, H, W).
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape((x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.

        Args:
            x (torch.Tensor): Input tensor of shape (N, L, D).
            mask_ratio (float): Ratio of values to mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Masked input, binary mask, shuffled indices.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            mask_ratio (float): Ratio of values to mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Encoded representation, binary mask, shuffled indices.
        """
        x = self.patchify(x)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (N, L, D).
            ids_restore (torch.Tensor): Indices to restore the original order of patches.

        Returns:
            torch.Tensor: Decoded output tensor of shape (N, L, patch_size^2 * in_chans).
        """
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Calculate the loss.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).
            pred (torch.Tensor): Predicted output of shape (N, L, patch_size^2 * in_chans).
            mask (torch.Tensor): Binary mask of shape (N, L).

        Returns:
            torch.Tensor: Computed loss value.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """
        Forward pass.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).
            mask_ratio (float): Ratio of values to mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Loss value, predicted output, binary mask.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (Tuple[torch.Tensor]): Input batch of images and corresponding labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss value for the current step.
        """
        imgs, _ = batch
        loss, _, _ = self(imgs)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (Tuple[torch.Tensor]): Input batch of images and corresponding labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss value for the current step.
        """
        imgs, _ = batch
        loss, _, _ = self(imgs)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def configure_optimizers(self):
        """
        Configure optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Define model architectures
mae_vit_small_patch16 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=768,
    depth=6,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=4,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

mae_vit_base_patch16 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

mae_vit_large_patch16 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

mae_vit_huge_patch14 = partial(
    MaskedAutoencoderViT,
    patch_size=14,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

mae_vit_large_patch16D4d256 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    decoder_embed_dim=256,
    decoder_depth=4,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

mae_vit_base_patch16D4d256 = partial(
    MaskedAutoencoderViT,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=256,
    decoder_depth=4,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)


import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import numpy as np


def main():
    # Create random data
    N = 32  # Batch size
    C, H, W = 3, 224, 224  # Image dimensions
    img_data = np.random.rand(N, C, H, W).astype(np.float32)
    target_data = np.random.randint(0, 10, size=N)  # Random labels
    imgs = torch.tensor(img_data)
    targets = torch.tensor(target_data)

    # Create a Lightning DataModule
    class RandomDataModule(pl.LightningDataModule):
        def __init__(self, imgs, targets, batch_size=32):
            super().__init__()
            self.imgs = imgs
            self.targets = targets
            self.batch_size = batch_size

        def train_dataloader(self):
            dataset = TensorDataset(self.imgs, self.targets)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        def val_dataloader(self):
            dataset = TensorDataset(self.imgs, self.targets)
            return DataLoader(dataset, batch_size=self.batch_size)

    # Instantiate Lightning DataModule
    data_module = RandomDataModule(imgs, targets)

    # Instantiate the model
    model = MaskedAutoencoderViT()

    # Instantiate the Lightning Trainer
    trainer = pl.Trainer(max_epochs=5)

    # Perform a forward pass
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
