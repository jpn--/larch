from .model import NumbaModel as Model
from .. import DataFrames, P, X, PX, OMX, DBF, Reporter, NumberedCaption, read_metadata, examples, util, __version__
from ..examples import example as _example
from ..data_warehouse import example_file
from ..model.model_group import ModelGroup

try:
    from ..dataset import Dataset, DataArray, DataTree, merge
except RuntimeError:
    pass

def example(*args, **kwargs):
    import importlib
    kwargs['larch'] = importlib.import_module(__name__)
    return _example(*args, **kwargs)
