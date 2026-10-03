"""错误码体系 E100~E500（不变量破坏 / 配置 / 数据 / 库 / 拟合失败）。"""


class SindyError(Exception):
    code = "E000"
    """基类错误。"""


class ConfigError(SindyError):
    code = "E100"


class DataError(SindyError):
    code = "E200"


class LibraryError(SindyError):
    code = "E300"


class FitError(SindyError):
    code = "E400"


class DeterminismError(SindyError):
    code = "E500"
