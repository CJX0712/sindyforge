"""preprocess 层：可选状态缩放（默认恒等）。"""

from .normalize import NoopScaler, Standardizer, build_preprocessor

__all__ = ["NoopScaler", "Standardizer", "build_preprocessor"]
