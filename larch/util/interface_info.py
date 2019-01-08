
from .. import __version__
import sys, os

class Info:

    def __init__(self, appname='Larch', extra=True, version=None, path=None):
        self.appname = appname
        self.extra = extra
        self.version = version or __version__
        from .. import __path__
        self.path = path or __path__[0]


    def __repr__(self):
        r = (f"┌── {self.appname.upper()} {self.version} " + "─" * (57 - len(self.version)))
        v = '\n│'.join(sys.version.split('\n'))
        r += (f"\n│Python {v}")
        r += (f"\n│EXE ─ {sys.executable}")
        r += (f"\n│CWD ─ {os.getcwd()}" )
        for p in sys.path[:1]:
            r += (f"\n│PTH ┬ {p}")
        for p in sys.path[1:-1]:
            r += (f"\n│    ├ {p}")
        for p in sys.path[-1:]:
            r += (f"\n│    └ {p}")
        r += ("\n└───────────────────────────────────────────────────────────────────────────")
        return r

    def _repr_html_(self):
        from xmle import Elem
        xsign = Elem("div", {'class': 'larch_head_tag'})
        from .images import favicon
        p = xsign.elem('p', {'style': 'float:left;margin-top:6px'})
        p.elem('img', {
            'width': "32",
            'height': "32",
            'src': "data:image/png;base64,{}".format(favicon),
            'style': 'float:left;position:relative;top:-3px;padding-right:0.2em;'
        }, tail=f" {self.appname} ")
        p.elem('span', {'class': 'larch_head_tag_ver'}, text=f" {self.version} ")
        p.elem('span', {'class': 'larch_head_tag_pth'}, text=f" {self.path} ")
        from .images import camsyslogo_element
        xsign << camsyslogo_element
        if 'larch' in sys.modules:
            from .images import georgiatechlogo_element
            xsign << georgiatechlogo_element

        if self.extra:
            v = '\n│'.join(sys.version.split('\n'))
            xsign.elem('br')
            xinfo = xsign.elem('div', {'class': 'larch_head_tag_more', 'style':'margin-top:10px; padding:7px'}, text=f'Python {v}')
            xinfo.elem('br', tail=f"EXE - {sys.executable}")
            xinfo.elem('br', tail=f"CWD - {os.getcwd()}")
            xinfo.elem('br', tail=f"PATH - ")
            ul = xinfo.elem('ul', {'style': 'margin-top:0; margin-bottom:0;'})
            for p in sys.path:
                ul.elem('li', text=p)

        from ..util.styles import _default_css_jupyter, _tooltipped_style_css
        style_prefix = "<style>{}\n\n{}</style>\n".format(_default_css_jupyter, _tooltipped_style_css)

        return style_prefix+xsign.tostring()


    def __call__(self, *args, **kwargs):
        """
        Calling an Info object is a no-op.

        Returns
        -------
        self
        """
        return self




def ipython_status(magic_matplotlib=False):
    message_set = set()
    try:
        # This will work in iPython, and fail otherwise
        cfg = get_ipython().config
    except:
        message_set.add('Not IPython')
    else:
        import IPython
        import IPython.core.error
        message_set.add('IPython')

        if magic_matplotlib:
            try:
                get_ipython().magic("matplotlib inline")
            except (IPython.core.error.UsageError, KeyError):
                message_set.add('IPython inline plotting not available')

        # Caution: cfg is an IPython.config.loader.Config
        if cfg['IPKernelApp']:
            message_set.add('IPython QtConsole')
            try:
                if cfg['IPKernelApp']['pylab'] == 'inline':
                    message_set.add('pylab inline')
                else:
                    message_set.add('pylab loaded but not inline')
            except:
                message_set.add('pylab not loaded')
        elif cfg['TerminalIPythonApp']:
            try:
                if cfg['TerminalIPythonApp']['pylab'] == 'inline':
                    message_set.add('pylab inline')
                else:
                    message_set.add('pylab loaded but not inline')
            except:
                message_set.add('pylab not loaded')
    return message_set

