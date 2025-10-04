from typing import Any


def convert_to_flat_dict(
    nested_dict: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """
    Преобразует вложенный словарь в плоский.
    Пример: {'a': 1, 'b': {'c': 2}} -> {'a': 1, 'b.c': 2}
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(convert_to_flat_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flattened_dict_to_nested(flattened_dict: dict[str, Any], sep: str = "."):
    """
    Преобразует плоский словарь обратно во вложенный.
    Пример: {'a': 1, 'b.c': 2} -> {'a': 1, 'b': {'c': 2}}
    """
    result = {}
    for key, value in flattened_dict.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]

        d[parts[-1]] = value
    return result
