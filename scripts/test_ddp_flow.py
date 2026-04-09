"""Minimal DDP test for flow training."""
import torch
from lam.model import LAM
from lam.dataset import LightningVideoDataset
from lightning import Trainer

dm = LightningVideoDataset(
    batch_size=4, num_workers=4,
    dataset_paths=['/media/data1/chan/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot'],
    flow_dir='/media/data1/chan/GR1_optical_flow',
    padding='repeat', randomize=True, num_frames=2,
    output_format='t h w c', samples_per_epoch=100, sampling_strategy='dataset',
)

model = LAM(
    image_channels=3, flow_channels=2,
    lam_model_dim=768, lam_latent_dim=32, lam_patch_size=16,
    lam_enc_blocks=16, lam_dec_blocks=16, lam_num_heads=12,
    beta=1e-6
)

trainer = Trainer(
    max_epochs=1, accelerator='gpu', devices=2,
    strategy='ddp_find_unused_parameters_false',
    precision='16-mixed', gradient_clip_val=0.3,
    accumulate_grad_batches=2,
    log_every_n_steps=10,
    enable_checkpointing=False, logger=False,
)
trainer.fit(model, dm)
print('DDP FLOW TRAINING OK')
