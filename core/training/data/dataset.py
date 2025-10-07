import random
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from core.nn.text_encoder import NewsTokenizerWrapper


class TimesFMDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer_path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix != ".arrow":
            raise ValueError(f"Expected an Arrow file, got {path.suffix}")

        self.df: pl.DataFrame | None = None
        self.news_tokenizer: NewsTokenizerWrapper | None = None
        self.path = path
        self.tokenizer_path = tokenizer_path

        self.output_patch_len = 128
        self.input_patch_len = 32
        self.slice_len = 256
        self.num_slices_per_ticker = 40
        self.idx2ticker = self.prepare()
        return

    def prepare(self) -> dict[int, str]:
        df = self._load_df()
        tickers = df["ticker"].unique().sort().to_list()
        idx2ticker = {i: t for i, t in enumerate(tickers)}
        del df
        return idx2ticker

    def _load_df(self) -> pl.DataFrame:
        df = pl.read_ipc(self.path, memory_map=True).sort("begin")
        return df

    def _init_worker(self) -> None:
        self.df = self._load_df()
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.news_tokenizer = NewsTokenizerWrapper(tokenizer, warn=False)
        return

    def __len__(self) -> int:
        return len(self.idx2ticker) * self.num_slices_per_ticker

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        if (self.df is None) or (self.news_tokenizer is None):
            self._init_worker()

        ticker_idx, _ = divmod(index, self.num_slices_per_ticker)
        ticker = self.idx2ticker[ticker_idx]
        slice_len = self.slice_len

        ticker_df = self.df.filter(pl.col("ticker") == ticker).sort("begin")

        start_idx = random.randint(
            0, max(0, len(ticker_df) - slice_len - 1 - self.output_patch_len)
        )
        end_idx = start_idx + slice_len + self.output_patch_len

        slice_df = ticker_df[start_idx:end_idx]

        ts = slice_df["close"].to_torch().float()

        tokens = slice_df["tokenized"].to_list()[:slice_len]

        inputs_ts = ts[:slice_len]
        mask_ts = torch.zeros_like(inputs_ts)
        if (pad_len := -(len(inputs_ts)) % self.input_patch_len) != 0:
            inputs_ts = torch.cat(
                [
                    torch.zeros(
                        pad_len,
                        device=inputs_ts.device,
                    ),
                    inputs_ts,
                ],
                dim=0,
            )
            mask_ts = torch.cat(
                [
                    torch.ones(
                        pad_len,
                        device=mask_ts.device,
                        dtype=torch.bool,
                    ),
                    mask_ts,
                ],
                dim=0,
            )
        patch_count = len(inputs_ts) // self.input_patch_len
        targets = torch.zeros(
            patch_count, self.output_patch_len, device=inputs_ts.device
        )
        texts_per_tokens: list[torch.Tensor] = []
        for i in range(1, patch_count + 1):
            targets[i - 1] = ts[
                i * self.input_patch_len : i * self.input_patch_len
                + self.output_patch_len
            ]

            texts_per_token = tokens[
                (i - 1) * self.input_patch_len : i * self.input_patch_len
            ]
            formatted_tokens = self.news_tokenizer.apply_format_to_tokens(
                texts_per_token
            )
            texts_per_tokens.append(formatted_tokens)
        texts_per_tokens_t = torch.nn.utils.rnn.pad_sequence(
            texts_per_tokens,
            batch_first=True,
            padding_value=self.news_tokenizer.tokenizer.pad_token_id,  # type: ignore
        )
        return {
            "inputs_ts": inputs_ts,
            "targets": targets,
            "mask_ts": mask_ts,
            "texts_per_tokens": texts_per_tokens_t,
        }


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    text_pad_token_id: int,
    text_model_max_length: int,
    text_sep_token_id: int,
) -> dict[str, torch.Tensor]:
    inputs_ts = torch.stack([item["inputs_ts"] for item in batch], dim=0)
    mask_ts = torch.stack([item["mask_ts"] for item in batch], dim=0)
    targets = torch.stack([item["targets"] for item in batch], dim=0)

    texts = [item["texts_per_tokens"] for item in batch]
    max_L = min(max(t.size(-1) for t in texts), text_model_max_length)
    texts_clamped: list[torch.Tensor] = []
    for t in texts:
        # оставляем правую часть длиной max_L
        if t.size(-1) > max_L:
            t = t[..., -(max_L - 1) :]
            t = torch.cat([t.new_full((t.size(0), 1), text_sep_token_id), t], dim=-1)
        # добиваем слева PAD токенами до max_L
        elif t.size(-1) < max_L:
            pad_len = max_L - t.size(-1)
            pad = torch.full(
                (t.size(0), pad_len), text_pad_token_id, dtype=t.dtype, device=t.device
            )
            t = torch.cat([pad, t], dim=-1)
        texts_clamped.append(t)

    inputs_text = torch.nn.utils.rnn.pad_sequence(
        texts_clamped,
        batch_first=True,
        padding_value=text_pad_token_id,
    )
    mask_text = (inputs_text != text_pad_token_id).long()
    return {
        "inputs_ts": inputs_ts,
        "mask_ts": mask_ts,
        "targets": targets,
        "inputs_text": inputs_text.long(),
        "mask_text": mask_text.long(),
    }
