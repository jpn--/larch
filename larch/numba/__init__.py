from .model import NumbaModel as Model
from .. import DataFrames, P, X, PX, OMX, DBF, Reporter, NumberedCaption, read_metadata, examples, util
from ..examples import example as _example
try:
    from ..dataset import Dataset, DataPool, DataArray
except RuntimeError:
    pass

def example(*args, **kwargs):
    import importlib
    kwargs['larch'] = importlib.import_module(__name__)
    return _example(*args, **kwargs)
