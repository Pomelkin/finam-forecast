from functools import partial
from pathlib import Path

import lightning as L
from clearml import Dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .cache import prepare_dataset_cache
from .dataset import collate_fn
from .dataset import TimesFMDataset
from core.nn.text_encoder import NewsTokenizerWrapper
from core.training.configs import DataConfig
from core.utils import setup_logger

logger = setup_logger()


class TimesFMDataModule(L.LightningDataModule):
    def __init__(
        self,
        path: str | Path,
        data_cfg: DataConfig | dict,
        tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        super().__init__()
        if isinstance(data_cfg, dict):
            data_cfg = DataConfig.model_validate(data_cfg)

        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        self.data_cfg = data_cfg

        self.clearml_dataset = Dataset.get(
            data_cfg.dataset_id, alias="Training Dataset"
        )

        self.batch_size = data_cfg.batch_size
        self.num_workers = data_cfg.num_workers

        self.cache_path: None | Path = None
        self.train_dataset: TimesFMDataset | None = None
        self.val_dataset: TimesFMDataset | None = None
        self.news_tokenizer = NewsTokenizerWrapper(tokenizer)
        return

    def prepare_data(self) -> None:
        prepare_dataset_cache(self.clearml_dataset)
        return

    def _dataset_factory(self, path: str | Path) -> TimesFMDataset:
        return TimesFMDataset(path=path, news_tokenizer=self.news_tokenizer)

    def setup(self, stage: str) -> None:
        if self.cache_path is None:
            self.cache_path = prepare_dataset_cache(self.clearml_dataset)
        if stage == "fit":
            if self.train_dataset is None:
                self.train_dataset = self._dataset_factory(
                    path=self.cache_path / "train.arrow"
                )
            if self.val_dataset is None:
                self.val_dataset = self._dataset_factory(
                    path=self.cache_path / "val.arrow",
                )
        elif stage == "validate":
            if self.val_dataset is None:
                self.val_dataset = self._dataset_factory(
                    path=self.cache_path / "val.arrow",
                )
        return

    @property
    def dataset_tokenizer(self) -> PreTrainedTokenizerFast:
        if self.train_dataset is not None:
            return self.train_dataset.news_tokenizer.tokenizer
        elif self.val_dataset is not None:
            return self.val_dataset.news_tokenizer.tokenizer
        else:
            raise ValueError("Datasets are not initialized, cannot get tokenizer.")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.train_dataset is None:
            raise ValueError("Training dataset is not initialized.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=partial(
                collate_fn,
                text_pad_token_id=self.dataset_tokenizer.pad_token_id,  # type: ignore
                text_model_max_length=self.dataset_tokenizer.model_max_length,
            ),  # type: ignore
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=partial(
                collate_fn,
                text_pad_token_id=self.dataset_tokenizer.pad_token_id,  # type: ignore
                text_model_max_length=self.dataset_tokenizer.model_max_length,
            ),  # type: ignore
        )
