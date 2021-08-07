
__version__ = '5.5.6'

from .util.interface_info import Info, ipython_status
import sys
from .util.styles import css

info = Info('Larch', False, __version__)
version = Info('Larch', False, __version__, minimal=True)


def require_version(n):
    from . import __version__
    try:
        from packaging import version
    except:
        def int_from(x):
            import re
            nums = re.findall(r'\d+', x)
            if nums:
                return int(nums[0])
            return 0
        r = [int_from(i) for i in n.split(".")[:2]]
        v = [int_from(i) for i in __version__.split(".")[:2]]
        if v[0] > r[0]:
            return
        if v[0] < r[0]:
            raise ValueError("the installed larch is version {}".format(__version__))
        if len(r)>=2:
            if v[1] > r[1]:
                return
            if v[1] < r[1]:
                raise ValueError("the installed larch is version {}".format(__version__))
        if len(r)>=3:
            if v[2] > r[2]:
                return
            if v[2] < r[2]:
                raise ValueError("the installed larch is version {}".format(__version__))
    else:
        if version.parse(n) > version.parse(__version__):
            raise ValueError("the installed larch is version {}".format(__version__))


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
