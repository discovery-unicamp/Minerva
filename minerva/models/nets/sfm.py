from functools import partial

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from minerva.utils.position_embedding import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(L.LightningModule):
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
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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

    def patchify(self, imgs):  # input: (32, 1, 224, 224)
        """
        Extract patches from input images.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).

        Returns:
            torch.Tensor: Patches of shape (N, num_patches, patch_size^2 * in_chans).
        """
        p = self.patch_embed.patch_size[0]
        assert (
            imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        )  # only square images are supported, and the size must be divisible by the patch size

        h = w = imgs.shape[2] // p
        x = imgs.reshape(
            (imgs.shape[0], self.in_chans, h, p, w, p)
        )  # Transform images into (32, 1, 14, 16, 14, 16)
        x = torch.einsum("nchpwq->nhwpqc", x)  # reshape into (32, 14, 14, 16, 16, 1)
        x = x.reshape(
            (imgs.shape[0], h * w, p**2 * self.in_chans)
        )  # Transform into (32, 196, 256)
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
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

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
        x = self.patch_embed(x)
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

# mae_vit_small_patch16_dec512d8b
# decoder: 512 dim, 8 blocks, depth: 6
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

# mae_vit_base_patch16_dec512d8b
# decoder: 512 dim, 8 blocks,
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

# mae_vit_large_patch16_dec512d8b
# decoder: 512 dim, 8 blocks
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

# mae_vit_huge_patch14_dec512d8b
# decoder: 512 dim, 8 blocks
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

# mae_vit_large_patch16_dec256d4b
# decoder: 256 dim, 8 blocks
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


# mae_vit_base_patch16_dec256d4b
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

# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import pytorch_lightning as pl
# import numpy as np


# def main():
#     # Create random data
#     N = 32  # Batch size
#     C, H, W = 1, 224, 224  # Image dimensions
#     img_data = np.random.rand(N, C, H, W).astype(np.float32)
#     target_data = np.random.randint(0, 10, size=N)  # Random labels
#     imgs = torch.tensor(img_data)
#     targets = torch.tensor(target_data)

#     # Create a Lightning DataModule
#     class RandomDataModule(L.LightningDataModule):
#         def __init__(self, imgs, targets, batch_size=32):
#             super().__init__()
#             self.imgs = imgs
#             self.targets = targets
#             self.batch_size = batch_size

#         def train_dataloader(self):
#             dataset = TensorDataset(self.imgs, self.targets)
#             return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         def val_dataloader(self):
#             dataset = TensorDataset(self.imgs, self.targets)
#             return DataLoader(dataset, batch_size=self.batch_size)

#     # Instantiate Lightning DataModule
#     data_module = RandomDataModule(imgs, targets)

#     # Instantiate the model
#     model = MaskedAutoencoderViT(img_size=224, patch_size=16, in_chans=1, embed_dim=256, depth=24, num_heads=16)

#     # Instantiate the Lightning Trainer
#     trainer = L.Trainer(max_epochs=5, devices=1, accelerator="cpu")

#     print("Forward pass...")
#     # Perform a forward pass
#     trainer.fit(model, datamodule=data_module)


# if __name__ == "__main__":
#     main()
