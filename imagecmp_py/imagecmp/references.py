"""Reference-image discovery and Unicode-safe local image loading.

The internal case layout may contain one primary reference image and several
additional references.  All references belong to one case and share the
case's standard-image YOLO ROI file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


LIVE_IMAGE_FILENAME = "对比截图.jpg"
PRIMARY_REFERENCE_FILENAME = "标准源图.jpg"
ROI_FILENAME = "标准源图坐标.txt"
_ADDITIONAL_REFERENCE_PATTERN = re.compile(r"^新增标准源图(\d+)\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class ReferenceImage:
    """One candidate reference image for a case."""

    reference_id: str
    path: Path


@dataclass(frozen=True)
class CaseInput:
    """Local files required to process one multi-reference case directory."""

    case_directory: Path
    live_path: Path
    roi_path: Path
    references: tuple[ReferenceImage, ...]


def read_color_image(path: Path) -> np.ndarray | None:
    """Read an image through byte input so Windows Unicode paths work."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def discover_case_input(case_directory: Path) -> CaseInput:
    """Discover the live image, ROI file, and all ordered reference images.

    Expected local layout::

        对比截图.jpg
        标准源图.jpg
        新增标准源图0.jpg
        新增标准源图1.jpg
        标准源图坐标.txt

    Additional reference indices must be contiguous from zero.  A malformed
    case is an input error rather than an unavailable-detection outcome.
    """
    case_directory = Path(case_directory)
    if not case_directory.is_dir():
        raise FileNotFoundError(f"case directory not found: {case_directory}")

    live_path = case_directory / LIVE_IMAGE_FILENAME
    roi_path = case_directory / ROI_FILENAME
    primary_path = case_directory / PRIMARY_REFERENCE_FILENAME
    missing = [
        name for name, path in (
            (LIVE_IMAGE_FILENAME, live_path),
            (ROI_FILENAME, roi_path),
            (PRIMARY_REFERENCE_FILENAME, primary_path),
        ) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"case {case_directory.name!r} is missing required file(s): {', '.join(missing)}"
        )

    additional: list[tuple[int, Path]] = []
    for path in case_directory.iterdir():
        if not path.is_file():
            continue
        match = _ADDITIONAL_REFERENCE_PATTERN.fullmatch(path.name)
        if match:
            additional.append((int(match.group(1)), path))
    additional.sort(key=lambda item: item[0])
    indices = [index for index, _ in additional]
    if indices != list(range(len(indices))):
        raise ValueError(
            f"additional reference indices in {case_directory.name!r} must start at 0 and be contiguous"
        )

    references = [ReferenceImage("primary", primary_path)]
    references.extend(
        ReferenceImage(f"additional_{index}", path)
        for index, path in additional
    )
    return CaseInput(
        case_directory=case_directory,
        live_path=live_path,
        roi_path=roi_path,
        references=tuple(references),
    )
