from functools import partial
from pathlib import Path
from typing import Any

import lightning as L
from clearml import Dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from .cache import prepare_dataset_cache
from core.training.configs import DataConfig
from core.utils import setup_logger

logger = setup_logger(Path(__file__).name)


class HintClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_cfg: DataConfig | dict[str, Any],
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()
        if isinstance(data_cfg, dict):
            data_cfg = DataConfig.model_validate(data_cfg)

        self.save_hyperparameters({"data_cfg": data_cfg.model_dump()})

        self.data_cfg = data_cfg

        self.clearml_dataset = Dataset.get(
            data_cfg.dataset_id, alias="Training Dataset"
        )
        self.batch_size = data_cfg.batch_size
        self.num_workers = data_cfg.num_workers
        self.tokenizer = tokenizer
        self.data_columns = data_cfg.data_columns
        self.label_column = data_cfg.label_column

        self.prompt_bundle = prompt_bundle_factory(
            prompt_type=data_cfg.prompt_type,
            add_system_prompt=data_cfg.add_system_prompt,
        )
        self.use_chat_template = data_cfg.use_chat_template

        self.cache_path: None | Path = None

        self.train_dataset: HintClassificationDataset | None = None
        self.val_dataset: HintClassificationDataset | None = None
        self.test_dataset: HintClassificationDataset | None = None
        self.predict_dataset: HintClassificationDataset | None = None
        return

    def prepare_data(self) -> None:
        prepare_dataset_cache(self.clearml_dataset)
        return

    def _dataset_factory(
        self, path: str | Path, is_train: bool
    ) -> HintClassificationDataset:
        return HintClassificationDataset(
            path=path,
            data_columns=self.data_columns,
            label_column=self.label_column,
            tokenizer=self.tokenizer,
            prompt_bundle=self.prompt_bundle,
            use_chat_template=self.use_chat_template,
            aug_cfg=self.data_cfg.augmentations if is_train else None,
        )

    def setup(self, stage: str) -> None:
        if self.cache_path is None:
            self.cache_path = prepare_dataset_cache(self.clearml_dataset)
        if stage == "fit":
            if self.train_dataset is None:
                self.train_dataset = self._dataset_factory(
                    path=self.cache_path / "train" / "train.arrow",
                    is_train=True,
                )
            if self.val_dataset is None:
                self.val_dataset = self._dataset_factory(
                    path=self.cache_path / "val" / "val.arrow",
                    is_train=False,
                )
        elif stage == "validate":
            if self.val_dataset is None:
                self.val_dataset = self._dataset_factory(
                    path=self.cache_path / "val" / "val.arrow",
                    is_train=False,
                )
        elif stage == "test":
            if self.test_dataset is None:
                self.test_dataset = self._dataset_factory(
                    path=self.cache_path / "test" / "test.arrow",
                    is_train=False,
                )
        elif stage == "predict":
            if self.predict_dataset is None:
                self.predict_dataset = self._dataset_factory(
                    path=self.cache_path / "predict" / "predict.arrow",
                    is_train=False,
                )
        return

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
                pad_token_id=self.tokenizer.pad_token_id,  # type: ignore
                max_model_length=self.tokenizer.model_max_length,
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
                pad_token_id=self.tokenizer.pad_token_id,  # type: ignore
                max_model_length=self.tokenizer.model_max_length,
            ),  # type: ignore
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataset is None:
            raise ValueError("Test dataset is not initialized.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=partial(
                collate_fn,
                pad_token_id=self.tokenizer.pad_token_id,  # type: ignore
                max_model_length=self.tokenizer.model_max_length,
            ),  # type: ignore
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.predict_dataset is None:
            raise ValueError("Predict dataset is not initialized.")
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=partial(
                collate_fn,
                pad_token_id=self.tokenizer.pad_token_id,  # type: ignore
                max_model_length=self.tokenizer.model_max_length,
            ),  # type: ignore
        )
