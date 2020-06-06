
__version__ = '5.4.2'

from .util.interface_info import Info, ipython_status
import sys
from .util.styles import css

info = Info('Larch', False, __version__)
version = Info('Larch', False, __version__, minimal=True)

if 'IPython' in ipython_status():
    from .util.display import display
    try:
        # from .util.styles import stylesheet
        # stylesheet()
        # display(info)
        pass
    except:
        # print(repr(info))
        jupyter_active = False
    else:
        jupyter_active = True
else:
    jupyter_active = False
    # print(repr(info))


from .roles import P, X, PX
from .data_services import DataService
from .omx import OMX
from .data_services.dbf.dbf_reader import DBF

from .model import Model
from .dataframes import DataFrames

from .examples import example
from .util import figures
from .util.excel import ExcelWriter

_doctest_mode_ = False

_larch_self = sys.modules[__name__]

from xmle import Reporter, NumberedCaption
from xmle import load_metadata as read_metadata
from .workspace import make_reporter
