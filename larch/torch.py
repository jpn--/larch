
try:
    from _larch_torch import *
except ImportError:
    import warnings
    warnings.warn("larch.torch is not installed")

