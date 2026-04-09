from os import makedirs, path
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2 as cv
import numpy as np
import piq
import torch
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

OptimizerCallable = Callable[[Iterable], Optimizer]

from lam.modules import LatentActionModel


class LAM(LightningModule):
    def __init__(
        self,
        image_channels: int = 3,
        flow_channels: int = 0,
        # Latent action autoencoder
        lam_model_dim: int = 512,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 8,
        lam_dec_blocks: int = 8,
        lam_num_heads: int = 8,
        lam_dropout: float = 0.0,
        beta: float = 0.01,
        log_interval: int = 1000,
        log_path: str = "log_imgs",
        optimizer: OptimizerCallable = AdamW,
        ckpt_path: Optional[str] = None
    ) -> None:
        super(LAM, self).__init__()
        self.flow_mode = flow_channels > 0
        self.lam = LatentActionModel(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout,
            flow_channels=flow_channels
        )
        self.beta = beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer

        self.save_hyperparameters()

        if ckpt_path is not None:
            self.reload_ckpt(ckpt_path)

    def reload_ckpt(self, ckpt_path: str) -> None:
        if path.exists(ckpt_path):
            lam = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = self.load_state_dict(lam, assign=True)
            print(f"Restored LAM from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing LAM keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected LAM keys: {unexpected}")
        else:
            print(f"LAM checkpoint {ckpt_path} does not exist")

    def shared_step(self, batch: Dict) -> Tuple:
        outputs = self.lam(batch)
        kl_loss = -0.5 * torch.sum(1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp(), dim=1).mean()

        if self.flow_mode:
            gt_flow = batch["flow"]  # (B, 1, H, W, 2)
            pred_flow = outputs["flow_pred"]  # (B, 1, H, W, 2)
            mask = batch.get("flow_mask")  # (B, 1, H, W) bool or None

            sq_err = (gt_flow - pred_flow) ** 2  # (B, 1, H, W, 2)
            epe_per_pixel = torch.norm(gt_flow - pred_flow, p=2, dim=-1)  # (B, 1, H, W)

            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()  # (B, 1, H, W, 1)
                num_valid = mask_f.sum().clamp(min=1.0)
                mse_loss = (sq_err * mask_f).sum() / (num_valid * 2)  # avg over 2 channels
                epe = (epe_per_pixel * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
                mask_ratio = mask.float().mean()
            else:
                mse_loss = sq_err.mean()
                epe = epe_per_pixel.mean()
                mask_ratio = torch.tensor(1.0)

            loss = mse_loss + self.beta * kl_loss
            return outputs, loss, (
                ("mse_loss", mse_loss),
                ("kl_loss", kl_loss),
                ("epe", epe),
                ("mask_ratio", mask_ratio),
            )
        else:
            gt_future_frames = batch["videos"][:, 1:]
            mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
            loss = mse_loss + self.beta * kl_loss
            gt = gt_future_frames.clamp(0, 1).reshape(-1, *gt_future_frames.shape[2:]).permute(0, 3, 1, 2)
            recon = outputs["recon"].clamp(0, 1).reshape(-1, *outputs["recon"].shape[2:]).permute(0, 3, 1, 2)
            psnr = piq.psnr(gt, recon).mean()
            ssim = piq.ssim(gt, recon).mean()
            return outputs, loss, (
                ("mse_loss", mse_loss),
                ("kl_loss", kl_loss),
                ("psnr", psnr),
                ("ssim", ssim)
            )

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False
        )

        if batch_idx % self.log_interval == 0 and self.global_rank == 0:
            self.log_images(batch, outputs, "train")
        return loss

    # @torch.no_grad()
    # def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
    #     # Compute the validation loss
    #     outputs, loss, aux_losses = self.shared_step(batch)
    #
    #     # Log the validation loss
    #     self.log_dict(
    #         {**{"val_loss": loss}, **{f"val/{k}": v for k, v in aux_losses}},
    #         prog_bar=True,
    #         logger=True,
    #         on_step=True,
    #         on_epoch=True,
    #         sync_dist=True
    #     )
    #
    #     if batch_idx % self.log_interval == 0:  # Start of the epoch
    #         self.log_images(batch, outputs, "val")
    #     return loss

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log_images(batch, outputs, "test")
        return loss

    @staticmethod
    def flow_to_color(flow: np.ndarray) -> np.ndarray:
        """Convert optical flow (H, W, 2) to RGB visualization (H, W, 3)."""
        h, w, _ = flow.shape
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        ang = np.arctan2(flow[..., 1], flow[..., 0])
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2 + 180) % 180  # hue
        hsv[..., 1] = 255  # saturation
        max_mag = mag.max() + 1e-6
        hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)  # value
        return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

    def log_images(self, batch: Dict, outputs: Dict, split: str) -> None:
        if self.flow_mode:
            # Input frames: frame_t, frame_{t+1}
            frames = batch["videos"][0].clamp(0, 1).cpu().detach().numpy()  # (T, H, W, 3)
            frame_t = (frames[0] * 255).astype(np.uint8)  # (H, W, 3)
            frame_t1 = (frames[1] * 255).astype(np.uint8)  # (H, W, 3)
            # Flow visualizations
            gt_flow = batch["flow"][0, 0].cpu().detach().numpy()  # (H, W, 2)
            pred_flow = outputs["flow_pred"][0, 0].cpu().detach().numpy()  # (H, W, 2)
            gt_vis = self.flow_to_color(gt_flow)  # (H, W, 3)
            pred_vis = self.flow_to_color(pred_flow)  # (H, W, 3)
            # Layout: frame_t | frame_{t+1} | GT flow | pred flow
            compare = np.concatenate([frame_t, frame_t1, gt_vis, pred_vis], axis=1)  # (H, 4W, 3)
        else:
            gt_seq = batch["videos"][0].clamp(0, 1).cpu()
            recon_seq = outputs["recon"][0].clamp(0, 1).cpu()
            recon_seq = torch.cat([gt_seq[:1], recon_seq], dim=0)
            compare_seq = torch.cat([gt_seq, recon_seq], dim=1)
            compare = rearrange(compare_seq * 255, "t h w c -> h (t w) c")
            compare = compare.detach().numpy().astype(np.uint8)

        img_path = path.join(self.log_path, f"{split}_step{self.global_step:06}.png")
        makedirs(path.dirname(img_path), exist_ok=True)
        try:
            Image.fromarray(compare).save(img_path)
        except:
            pass

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim
