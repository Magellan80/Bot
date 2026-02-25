"""v30 compatibility adapter for SmartFilters v3.

Позволяет v30-коду импортировать SmartFilters из пространства имён `v30`.
"""

from smart_filters_v3 import SmartFilters, apply_smartfilters_v3, _GLOBAL_FILTERS

__all__ = [
    "SmartFilters",
    "apply_smartfilters_v3",
    "_GLOBAL_FILTERS",
]
