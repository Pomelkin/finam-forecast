from pathlib import Path
from typing import Literal

import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from core.utils import setup_logger


logger = setup_logger(fmt="only_message")


class NewsTokenizerWrapper:
    def __init__(
        self,
        tokenizer_path: str | Path,
        model_max_length: int = 4096,
        truncation_side: Literal["left", "right"] = "left",
        warn: bool = True,
    ) -> None:
        self.model_max_length = model_max_length
        self.truncation_side = truncation_side
        self.warn = warn
        self.tokenizer_path = tokenizer_path

        tokenizer = self._lazy_init_tokenizer()

        self.cls_token_id = int(tokenizer.cls_token_id)  # type: ignore
        self.sep_token_id = int(tokenizer.sep_token_id)  # type: ignore
        self.pad_token_id = int(tokenizer.pad_token_id)  # type: ignore
        self.tokenizer: PreTrainedTokenizerFast | None = None
        return

    def _lazy_init_tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        if tokenizer.model_max_length != self.model_max_length:
            if self.warn:
                logger.warning(
                    f"Tokenizer max length {tokenizer.model_max_length} is not equal to requested max_length {self.model_max_length}. Setting to {self.model_max_length}."
                )
            tokenizer.model_max_length = self.model_max_length

        if tokenizer.truncation_side != self.truncation_side:
            if self.warn:
                logger.warning(
                    f"Tokenizer truncation side {tokenizer.truncation_side} is not {self.truncation_side}. Setting to {self.truncation_side}."
                )
            tokenizer.truncation_side = self.truncation_side
        return tokenizer

    def tokenize_news(
        self, news: list[str | None], return_tensors: Literal["pt"] | None = None
    ) -> dict[str, list | torch.Tensor]:
        if self.tokenizer is None:
            self.tokenizer = self._lazy_init_tokenizer()

        token_ids: list[int] = []
        for publication in news:
            if publication is not None:
                # кодируем БЕЗ добавления спецтокенов (вставим sep самостоятельно)
                # используем add_special_tokens=False и берем только input_ids
                enc = self.tokenizer.encode(
                    publication, add_special_tokens=False, truncation=False
                )
                token_ids.extend(enc)
                token_ids.append(int(self.tokenizer.sep_token_id))  # type: ignore

        # обрезаем при необходимости (оставляем хвост контролируемой длины)
        max_body = max(0, int(self.tokenizer.model_max_length) - 1)
        if len(token_ids) > max_body:
            token_ids = token_ids[-max_body:]

        input_ids = [self.tokenizer.cls_token_id] + token_ids

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
            }
        else:
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    def apply_format_to_tokens(
        self, input_ids: list[list[list[int]] | None]
    ) -> torch.Tensor:
        formatted_inputs_ids: list[int] = []

        for doc_tokens in input_ids:
            if doc_tokens is not None:
                formatted_inputs_ids.extend(doc_tokens[0])
                formatted_inputs_ids.append(self.sep_token_id)  # type: ignore

        if len(formatted_inputs_ids) > (self.model_max_length - 1):
            formatted_inputs_ids = formatted_inputs_ids[-(self.model_max_length - 1) :]
        return torch.tensor(
            [self.cls_token_id] + formatted_inputs_ids, dtype=torch.long
        )
