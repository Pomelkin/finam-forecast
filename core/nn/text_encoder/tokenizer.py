from pathlib import Path
from typing import Literal

import torch
from transformers import AutoTokenizer

from ...utils import setup_logger


logger = setup_logger(__name__)


class NewsTokenizerWrapper:
    def __init__(
        self,
        pretrained_model_name_or_path: str | Path = "deepvk/RuModernBERT-base",
        max_length: int = 8192,
        truncation_side: Literal["left", "right"] = "left",
        no_news_token_id: int = 50285,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        if tokenizer.model_max_length != max_length:
            logger.warning(
                f"Tokenizer max length {tokenizer.model_max_length} is not equal to requested max_length {max_length}. Setting to {max_length}."
            )
            tokenizer.model_max_length = max_length

        if tokenizer.truncation_side != truncation_side:
            logger.warning(
                f"Tokenizer truncation side {tokenizer.truncation_side} is not {truncation_side}. Setting to {truncation_side}."
            )
            tokenizer.truncation_side = truncation_side

        self.tokenizer = tokenizer
        self.no_news_token = tokenizer.convert_ids_to_tokens(no_news_token_id)
        self.no_news_token_id = no_news_token_id
        return

    def tokenize_news(
        self, news: list[str | None], return_tensors: Literal["pt"] | None = None
    ) -> dict[str, list | torch.Tensor]:
        token_ids: list[int] = []
        for publication in news:
            if publication is None:
                token_ids.append(self.no_news_token_id)
            else:
                # кодируем БЕЗ добавления спецтокенов (вставим sep самостоятельно)
                # используем add_special_tokens=False и берем только input_ids
                enc = self.tokenizer.encode(
                    publication, add_special_tokens=False, truncation=False
                )
                token_ids.extend(enc)
            token_ids.append(self.tokenizer.sep_token_id)

        # обрезаем при необходимости (оставляем хвост контролируемой длины)
        max_body = max(0, int(self.tokenizer.model_max_length) - 1)
        if len(token_ids) > max_body:
            token_ids = token_ids[-max_body:]

        input_ids = [self.tokenizer.cls_token_id] + token_ids

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
            }
        else:
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    def apply_format_to_tokens(self, input_ids: list[list[int] | None]) -> list[int]:
        formatted_inputs_ids = []

        for doc_tokens in input_ids:
            if doc_tokens is None:
                formatted_inputs_ids.append(self.no_news_token_id)
            else:
                formatted_inputs_ids.extend(doc_tokens)
            formatted_inputs_ids.append(self.tokenizer.sep_token_id)

        if len(formatted_inputs_ids) > (self.tokenizer.model_max_length - 1):
            formatted_inputs_ids = formatted_inputs_ids[
                -(self.tokenizer.model_max_length - 1) :
            ]
        return [self.tokenizer.cls_token_id] + formatted_inputs_ids
