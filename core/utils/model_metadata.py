import re


def increment_version(s: str) -> str:
    """
    Increments the minor part of a version string:
    - 'vX.Y' or 'v.X.Y' (an optional dot after 'v' is supported)
    Examples:
    v1.00 -> v1.01
    v2.99 -> v2.100
    v.3.009 -> v.3.010
    """
    s = s.strip()
    m = re.fullmatch(r"v(\.?)(\d+)\.(\d+)", s)
    if not m:
        raise ValueError(
            f"Неверный формат версии: {s!r}. Ожидается 'vX.Y' или 'v.X.Y'."
        )

    vdot, major_str, minor_str = m.groups()
    major = int(major_str)
    minor = int(minor_str) + 1

    # сохраняем leading zeros по исходной ширине, длина может увеличиться (99 -> 100)
    minor_out = str(minor).zfill(len(minor_str))
    prefix = f"v{vdot}"  # 'v' или 'v.'
    return f"{prefix}{major}.{minor_out}"


def find_version_in_tags(tags: list[str]) -> str | None:
    """
    Finds the first version tag in the list of tags.
    Version tags are in the format 'vX.Y' or 'v.X.Y' (an optional dot after 'v' is supported).
    Examples:
    - 'v1.00'
    - 'v2.99'
    - 'v.3.009'
    """
    version_pattern = re.compile(r"^v(\.?)(\d+)\.(\d+)$")
    for tag in tags:
        if version_pattern.match(tag):
            return tag
    return None
