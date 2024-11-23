import numpy as np
import math
from typing import List, Union, Optional, Dict, Tuple

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import Metric

from torch.nn.modules.loss import CrossEntropyLoss

from scipy.ndimage.interpolation import zoom
# from einops import repeat
import gc

from . import sam_model_registry
from .util import DiceLoss, Focal_loss

class LoRA(nn.Module):
    def __init__(self, original_module, bias=True, alpha=1, r=1):
        super(LoRA, self).__init__()

        self.original_module = original_module
        self.matrix_A = torch.nn.Linear(original_module.in_features, r, bias=bias)
        self.matrix_B = torch.nn.Linear(r, original_module.out_features, bias=bias)
        self.scaling = alpha / r

        self.init_weights()
    
    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.matrix_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.matrix_B.weight)
        
    def forward(self, x):
        return self.original_module(x) + self.scaling * self.matrix_B(self.matrix_A(x))

class SAMLoRA(L.LightningModule):
    def __init__(self, 
                 image_size: int = 256,
                 num_classes: int = 5,
                 pixel_mean: List[int] = [0, 0, 0],
                 pixel_std: List[int] = [1, 1, 1],
                 alpha: int = 1, 
                 rank: int = 4, 
                 apply_lora_vision_encoder: bool = True,
                 apply_lora_mask_decoder: bool = True,
                 frozen_vision_encoder: bool = True,
                 frozen_prompt_encoder: bool = True,
                 frozen_mask_decoder: bool = True,
                 vit_model: str = "vit_b",
                 checkpoint = None,# str = "../checkpoints_sam/sam_vit_b_01ec64.pth",
                #  loss_fn: Optional[nn.Module] = None,
                 train_metrics: Optional[Dict[str, Metric]] = None,
                 val_metrics: Optional[Dict[str, Metric]] = None,
                 test_metrics: Optional[Dict[str, Metric]] = None):
        super(SAMLoRA, self).__init__()
        self.train_metrics_acc = []  # Armazenar métricas de treino
        self.train_loss_acc = []  # Armazenar losses de treino
        self.val_metrics_acc = []  # Armazenar métricas de validação
        self.val_loss_acc = []  # Armazenar losses de validação

        self.image_size = image_size

        # loss functions usadas
        # class_weight = torch.tensor([0.0456399, 0.1064931, 0.02634881, 0.1825596, 0.4259724, 0.2129862]).to(self.device, dtype=torch.float) # pesos (frequencia inversa das classes): do gabriel
        # minha frequencia coletada
        class_counts = torch.tensor([152539984, 63299037, 256085255, 37363084, 17793353, 6644487], dtype=torch.float)
        # Frequência inversa
        class_weights = 1.0 / class_counts # notou-se que aplicar assim fica melhor
        # Normalizar os pesos para evitar valores muito extremos
        # normalized_weights = class_weights / class_weights.sum() # notou-se que aplicar isso fica levemente ruim

        self.ce_loss = CrossEntropyLoss(weight=class_weights.to(self.device)) # weight=class_weights.to(self.device)
        self.dice_loss = DiceLoss(num_classes + 1)
        self.focal_loss = Focal_loss(num_classes=num_classes + 1)

        self.train_metrics = train_metrics or self._init_metrics()
        self.val_metrics = val_metrics or self._init_metrics()
        self.test_metrics = test_metrics or self._init_metrics()

        self.model, self.img_embedding_size = sam_model_registry[vit_model](
            image_size=image_size,
            num_classes=num_classes,
            checkpoint=checkpoint,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std
        )
        
        # frozen layers
        if frozen_vision_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if frozen_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if frozen_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        
        # apply lora
        if apply_lora_vision_encoder:
            self.__apply_lora_vision_encoder(alpha, rank)
        if apply_lora_mask_decoder:
            self.__apply_lora_mask_decoder(alpha, rank)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    def _init_metrics(self):
        return {
            'pixel_accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=6),
            'mean_class_accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=6, average='macro'),
            'dice_score': torchmetrics.Dice(num_classes=6),
            'mIoU': torchmetrics.JaccardIndex(task="multiclass", num_classes=6),
            # 'iou_per_class': torchmetrics.JaccardIndex(task="multiclass", num_classes=6, average=None),
        }
    
    # TODO: segundo o paper do LoRA, é suficiente aplicar só em uma das matrizes Q, K ou V, separadamente (testar isso)
    def __apply_lora_vision_encoder(self, alpha, rank):
        for layer in self.model.image_encoder.blocks:
            layer.attn.qkv = LoRA(original_module=layer.attn.qkv, bias=True, alpha=alpha, r=rank)
            # layer.mlp.lin1 = LoRA(original_module=layer.mlp.lin1, bias=True, alpha=alpha, r=rank)
            # layer.mlp.lin2 = LoRA(original_module=layer.mlp.lin2, bias=True, alpha=alpha, r=rank)
    
    # TODO: segundo o paper do LoRA, é suficiente aplicar só em uma das matrizes Q, K ou V, separadamente (testar isso)
    def __apply_lora_mask_decoder(self, alpha, rank):
        for layer in self.model.mask_decoder.transformer.layers:
            layer.self_attn.q_proj = LoRA(original_module=layer.self_attn.q_proj, bias=True, alpha=alpha, r=rank)
            # layer.self_attn.k_proj = LoRA(original_module=layer.self_attn.k_proj, bias=True, alpha=alpha, r=rank)
            layer.self_attn.v_proj = LoRA(original_module=layer.self_attn.v_proj, bias=True, alpha=alpha, r=rank)
            layer.cross_attn_token_to_image.q_proj = LoRA(original_module=layer.cross_attn_token_to_image.q_proj, bias=True, alpha=alpha, r=rank)
            # layer.cross_attn_token_to_image.k_proj = LoRA(original_module=layer.cross_attn_token_to_image.k_proj, bias=True, alpha=alpha, r=rank)
            layer.cross_attn_token_to_image.v_proj = LoRA(original_module=layer.cross_attn_token_to_image.v_proj, bias=True, alpha=alpha, r=rank)
    
    def forward(self, batched_input, multimask_output, image_size):
        return self.model(batched_input, multimask_output, image_size)
    
    def calc_loss(self, outputs, low_res_label_batch, dice_weight:float=0.8):
        low_res_logits = outputs['low_res_logits']
        loss_ce = self.ce_loss(low_res_logits, low_res_label_batch[:].long())
        loss_focal = self.focal_loss(low_res_logits, low_res_label_batch)
        loss_dice = self.dice_loss(low_res_logits, low_res_label_batch, softmax=True)
        
        loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
        # loss = (1 - dice_weight) * (0.5*loss_ce + 0.5*loss_focal) + dice_weight * loss_dice
        return loss, loss_ce, loss_dice, loss_focal
    
    def make_low_res_label(self, label, low_res=[32 * 4, 32 * 4], output_size=[512, 512]):
        # device = label.device
        _, label_h, label_w = label.shape

        # Mova o tensor para a CPU e converta-o em um array do NumPy antes de aplicar o zoom
        label = label.cpu().numpy()

        # if label_h != output_size[0] or label_w != output_size[1]:
        #     label = zoom(label, (output_size[0] / label_h, output_size[1] / label_w), order=0)
        low_res_label_np = zoom(label, (1, low_res[0] / label_h, low_res[1] / label_w), order=0)

        # Converta de volta para tensor e mova para o dispositivo original
        low_res_label = torch.from_numpy(low_res_label_np.astype(np.float32)).to(self.device)
        return low_res_label.long()

    def _step(self, batch, step_name):
        low_res = self.img_embedding_size * 4

        image_batch, label_batch = batch[0], batch[1]
        # label_batch = label_batch #.squeeze(1)  # Remove a dimensão extra
        low_res_label_batch = self.make_low_res_label(label=label_batch, low_res=[low_res, low_res]) # batch["low_res_label"]
        # debug
        # print('-'*20)
        # print(type(image_batch), image_batch.shape) # tem que ser (B C H W)
        # print(type(label_batch), label_batch.shape) # tem que ser (B H W)
        # print(low_res_label_batch.shape) # tem que ser (B low_resolution low_resolution)
        # print('-'*20)

        # assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
        
        outputs = self.model(image_batch, True, image_size=self.image_size)
        # print("outputs: ", outputs)
        # print("low_res_label_batch: ", low_res_label_batch)

        loss, loss_ce, loss_dice, loss_focal = self.calc_loss(outputs, low_res_label_batch)
        
        # Convert logits to probabilities for metrics
        probs = torch.softmax(outputs['masks'], dim=1)
        preds = torch.argmax(probs, dim=1).to(self.device) # forçando ficar no device
        label_batch = label_batch.to(self.device).long() # forçando ficar no device

        # # Update metrics
        # for metric in getattr(self, f"{step_name}_metrics").values():
        #     metric.to(self.device) # forçando ficar no device
        
        metrics = {}
        for metric_name, metric in getattr(self, f"{step_name}_metrics").items():
            metric.to(self.device).update(preds, label_batch)
            metrics[metric_name] = metric.compute().item()
            self.log(
                f"{step_name}_{metric_name}", 
                metrics[metric_name], 
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
        loss = self._step(batch, "train")
        self.train_loss_acc.append(loss.item())  # Armazena a loss do treino
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        loss = self._step(batch, "val")
        self.val_loss_acc.append(loss.item())  # Armazena a loss da validação
        return loss
    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._step(batch, "test")
    
    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None
    ):
        image_batch = batch[0]
        assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
        return self.model(image_batch, True, image_size=self.image_size)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.005, betas=(0.9, 0.999), weight_decay=0.1)
    
    def on_train_epoch_end(self):
        metrics = {}
        for metric_name, metric in self.train_metrics.items():
            # Compute e converte para valor numérico
            value = metric.compute().item()
            metrics[metric_name] = value
        avg_train_loss = sum(self.train_loss_acc) / len(self.train_loss_acc)
        metrics['loss'] = avg_train_loss
        self.train_loss_acc.clear()
        self.train_metrics_acc.append(metrics)

        for metric in self.train_metrics.values():
            metric.reset()
        
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        metrics = {}
        for metric_name, metric in self.val_metrics.items():
            # Compute e converte para valor numérico
            value = metric.compute().item()
            metrics[metric_name] = value
        avg_val_loss = sum(self.val_loss_acc) / len(self.val_loss_acc)
        metrics['loss'] = avg_val_loss
        self.val_loss_acc.clear()
        self.val_metrics_acc.append(metrics)

        for metric in self.val_metrics.values():
            metric.reset()
        
        gc.collect()
        torch.cuda.empty_cache()