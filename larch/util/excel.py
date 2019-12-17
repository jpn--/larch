
import numpy as np
import pandas as pd
from xmle import Elem
import matplotlib.figure
import io
import base64
from PIL import Image

try:
    import xlsxwriter
except ImportError:
    xlsxwriter = None
    _XlsxWriter = object
else:
    try:
        from pandas.io.excel._xlsxwriter import _XlsxWriter
    except ImportError:
        from pandas.io.excel import _XlsxWriter


class NumberedCaptions:

    def __init__(self, kind):
        self._kind = kind
        self._n = 1

    def __call__(self, caption):
        if caption:
            numb_caption = f"{self._kind} {self._n}: {caption}"
            self._n += 1
            return numb_caption
        else:
            return caption

class ExcelWriter(_XlsxWriter):
    engine = 'xlsxwriter_larch'
    supported_extensions = ('.xlsx',)

    def __init__(
            self,
            *args,
            model=None,
            data_statistics=True,
            nesting=True,
            numbering=True,
            hide_log=True,
            output_renderer=None,
            **kwargs,
    ):

        _engine = 'xlsxwriter_larch'
        kwargs.pop('engine', None)
        super().__init__(*args, engine=_engine, **kwargs)
        self.head_fmt = self.book.add_format({'bold': True, 'font_size':14})
        self.sheet_startrow = {}
        self._col_widths = {}
        if output_renderer is None:
            output_renderer = lambda x: x
        self._output_renderer = output_renderer

        if numbering:
            self.TAB = NumberedCaptions('Table')
            self.FIG = NumberedCaptions('Figure')
        else:
            self.TAB = lambda x: x
            self.FIG = lambda x: x

        self.add_worksheet('Parameters') # first sheet cannot be hidden

        self.logsheet = self.add_worksheet('_log_', hide=hide_log)
        self.log(f"larch.util.excel.ExcelWriter opened: {str(args)}")

        self.metadatasheet = self.add_worksheet('_metadata_', hide=True)
        self.metadatasheet.write(0, 0, 'key')
        self.metadatasheet.write(0, 1, 'len')
        self.metadatasheet.write(0, 2, 'val')
        self.sheet_startrow['_metadata_'] = 1

        if model is not None:
            self.add_model(model, data_statistics=data_statistics, nesting=nesting)


    def add_model(self, model, data_statistics=True, nesting=True):

        self.add_content_tab(model.pfo(), sheetname="Parameters", heading="Parameters" )

        if data_statistics:
            from .statistics import statistics_for_dataframe
            if model.dataframes.data_co is not None:
                self.add_content_tab(statistics_for_dataframe(model.dataframes.data_co), sheetname="CO Data", heading="CO Data")
            if model.dataframes.data_ca is not None:
                self.add_content_tab(statistics_for_dataframe(model.dataframes.data_ca), sheetname="CA Data", heading="CA Data")
            if model.dataframes.data_ce is not None:
                self.add_content_tab(statistics_for_dataframe(model.dataframes.data_ce), sheetname="CE Data", heading="CE Data")
            if model.dataframes.data_ch is not None and model.dataframes.data_av is not None:
                self.add_content_tab(model.dataframes.choice_avail_summary(graph=model.graph), sheetname="Choice", heading="Choices")

        if nesting and not model.is_mnl():
            self.add_content_tab(model.graph.__xml__(output='png'), sheetname="Nesting", heading="Nesting Tree")
            self.add_content_tab(model.graph_descrip('nodes'), sheetname="Nesting", heading="Nesting Node List")

    def add_metadata(self, key, value):
        if not isinstance(key, str):
            raise ValueError(f"metadata key must be str not {type(key)}")
        import base64
        try:
            import cloudpickle as pickle
        except ImportError:
            import pickle
        encoded_value = base64.standard_b64encode(pickle.dumps(value)).decode()
        row = self.sheet_startrow.get('_metadata_',0)
        self.metadatasheet.write(row, 0, key)
        c = 2
        while len(encoded_value):
            self.metadatasheet.write(row, c, encoded_value[:30_000]) # excel has cell text length cap
            c += 1
            encoded_value = encoded_value[30_000:]
        self.metadatasheet.write(row, 1, c-2)
        self.sheet_startrow['_metadata_'] = row+1

    def log(self, message):
        import time
        t = time.strftime("%Y-%b-%d %H:%M:%S")
        row = self.sheet_startrow.get('_log_',0)
        self.logsheet.write(row, 0, t)
        self.logsheet.write(row, 1, str(message))
        self.sheet_startrow['_log_'] = row+1

    def add_worksheet(self, name, force=False, hide=None):

        if not force and name in self.sheets:
            return self.sheets[name]

        s = None
        try:
            s = self.book.add_worksheet(name)
        except xlsxwriter.exceptions.DuplicateWorksheetName:
            i = 2
            while s is None:
                try:
                    s = self.book.add_worksheet(f"{name}{i}")
                except xlsxwriter.exceptions.DuplicateWorksheetName:
                    i += 1
                    if i > 99:
                        raise
        self.sheets[name] = s
        if hide is not None:
            if hide:
                s.hide()
            else:
                s.hidden = 0

        return s

    def add_content_tab(self, content, sheetname, heading=None, startrow=None):

        if startrow is None:
            startrow = self.sheet_startrow.get(sheetname, 0)
        worksheet = self.add_worksheet(sheetname)

        content_in = content
        success = False

        if isinstance(content, pd.DataFrame):
            if heading is not None:
                worksheet.write(startrow, 0, self.TAB(heading), self.head_fmt)
                startrow += 1
            content.to_excel(self, sheet_name=worksheet.name, startrow=2 if heading is not None else 0)
            startrow += len(content) + content.columns.nlevels
            if content.index.name:
                startrow += 1
            startrow += 2 # gap
            success = True

            if sheetname not in self._col_widths:
                self._col_widths[sheetname] = {}
            if content.index.nlevels == 1:
                current_width = self._col_widths[sheetname].get(0, 8)
                new_width = max(current_width, max(len(str(i)) for i in content.index))
                self._col_widths[sheetname][0] = new_width
                worksheet.set_column(0, 0, new_width)
            else:
                for n in range(content.index.nlevels):
                    current_width = self._col_widths[sheetname].get(n, 8)
                    new_width = max(current_width, max(len(str(i)) for i in content.index.levels[n]))
                    self._col_widths[sheetname][n] = new_width
                    worksheet.set_column(n,n,new_width)

        # Render matplotlib.Figure into an Elem
        if not success and 'matplotlib' in str(type(content)):
            if isinstance(content, matplotlib.figure.Figure):
                try:
                    content = Elem.from_figure(content, format='png', dpi='figure')
                except:
                    pass

        # Extract PNG data from Elem if found there
        if not success and isinstance(content, Elem):
            if content.tag == 'img':
                _v = content
            else:
                _v = content.find('img')
            if _v is not None:
                try:
                    _v = _v.attrib['src']
                except:
                    pass
                else:
                    if isinstance(_v, str) and _v[:22] == 'data:image/png;base64,':
                        _v = io.BytesIO(base64.standard_b64decode(_v[22:]))
                        img = Image.open(_v)
                        if heading is not None:
                            worksheet.write(startrow, 0, self.FIG(heading), self.head_fmt)
                            startrow += 1
                        worksheet.insert_image(startrow, 0, f'image-{sheetname}.png', {'image_data': _v})
                        startrow += int(np.ceil(img.size[1] / img.info.get('dpi',[96,96])[1] * 72 / 15)) + 3
                        success = True

        if not success:
            import warnings
            warnings.warn(f'content not written to sheet {sheetname}')

        self.sheet_startrow[worksheet.name] = startrow
        return self._output_renderer(content_in)

    def save(self, makedirs=True, overwrite=False):
        if makedirs:
            import os
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if not overwrite and not getattr(self, '__file_archived', False): # don't move twice
            from xmle.file_util import archive_existing_file
            try:
                new_name = archive_existing_file(self.path, archive_path=None, tag='creation')
            except FileNotFoundError:
                pass
            else:
                self.log(f"archived existing file to {new_name}")
                setattr(self, '__file_archived', new_name)

        self.log(f"saving")
        super().save()

if xlsxwriter is not None:
    from pandas.io.excel import register_writer
    register_writer(ExcelWriter)

def _make_excel_writer(model, filename, data_statistics=True):
    xl = ExcelWriter(filename, engine='xlsxwriter_larch', model=model, data_statistics=data_statistics)
    return xl

if xlsxwriter is not None:
    from .. import Model
    Model.to_xlsx = _make_excel_writer

def load_metadata(xlsx_filename, key=None):
    import pickle, base64, pandas
    raw = pandas.read_excel(xlsx_filename, sheet_name='_metadata_', index_col=0)
    metadata = {}
    if key is None:
        for key, row in raw.iterrows():
            v = (row[1:row[0] + 1].str.cat())
            metadata[key] = pickle.loads(base64.standard_b64decode(v.encode()))
        return metadata
    else:
        row = raw.loc[key]
        v = (row[1:row[0] + 1].str.cat())
        return pickle.loads(base64.standard_b64decode(v.encode()))

