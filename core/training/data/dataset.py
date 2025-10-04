from pathlib import Path

import polars as pl
from torch.utils.data import Dataset


class FinClipDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        ts_col_names: list[str],
        tokens_col_name: str,
        text_cls_token: int,
        text_sep_token: int,
        text_max_length: int,
        ts_seq_len: int,
        ts_num_channels: int,
    ) -> None:
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix != ".arrow":
            raise ValueError(f"Expected an Arrow file, got {path.suffix}")

        self.df: pl.DataFrame | None = None
        self.path = path
        self.ts_col_names = ts_col_names
        self.tokens_col_name = tokens_col_name

        self.text_cls_token = text_cls_token
        self.text_sep_token = text_sep_token
        self.text_max_length = text_max_length
        self.ts_seq_len = ts_seq_len
        self.ts_num_channels = ts_num_channels
        return

    def _load_df(self) -> pl.DataFrame:
        df = pl.read_ipc(
            self.path,
            memory_map=True,
            columns=self.ts_col_names + [self.tokens_col_name],
        )
        return df

    # def __getitem__(self, index) -> dict[str, torch.Tensor]:
    #     if self.df is None:
    #         self.df = self._load_df()

    #     row = self.df.row(index, named=True)

    #     tokens: list[int] = row[self.tokens_col_name]
    #     if len(tokens) > self.text_max_length - 2:
    #         tokens = tokens[: self.text_max_length - 2]
    #     tokens = [self.text_cls_token] + tokens + [self.text_sep_token]
    #     tokens_t = torch.tensor(tokens, dtype=torch.long)

    #     ts_seq_len: list[float] = row[self.ts_col_name]
