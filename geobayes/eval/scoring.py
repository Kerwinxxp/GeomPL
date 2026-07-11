"""距离评测支撑（层级地名组装）。

规范实现在 geobayes.eval.geocode（用户既定 API）；此处仅再导出，保持向后兼容。
"""
from .geocode import assemble_name, hierarchical_name  # noqa: F401
