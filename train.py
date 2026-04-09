from pathlib import Path

from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI

load_dotenv(Path(__file__).parent / ".env")

from lam.dataset import LightningVideoDataset
from lam.model import LAM

cli = LightningCLI(
    LAM,
    LightningVideoDataset,
    seed_everything_default=32,
    save_config_kwargs={"overwrite": True}
)
