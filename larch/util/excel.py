
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from xmle import Elem
import matplotlib.figure
import io
import base64
from .png import make_png

import logging

logger = logging.getLogger("Larch.Excel")

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
            embed_model=True,
            **kwargs,
    ):

        _engine = 'xlsxwriter_larch'
        kwargs.pop('engine', None)
        super().__init__(*args, engine=_engine, **kwargs)
        self.head_fmt = self.book.add_format({'bold': True, 'font_size':14})
        self.toc_link_fmt = self.book.add_format({'font_size':8})
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

        self.tocsheet = self.add_worksheet('Contents') # first sheet cannot be hidden
        self.tocsheet.write(0, 0, 'Table of Contents', self.head_fmt)
        self.sheet_startrow['Contents'] = 3

        self.logsheet = self.add_worksheet('_log_', hide=hide_log)
        self.log(f"larch.util.excel.ExcelWriter opened: {str(args)}")

        self.metadatasheet = self.add_worksheet('_metadata_', hide=True)
        self.metadatasheet.write(0, 0, 'key')
        self.metadatasheet.write(0, 1, 'len')
        self.metadatasheet.write(0, 2, 'val')
        self.sheet_startrow['_metadata_'] = 1

        if model is not None:
            self.add_model(model, data_statistics=data_statistics, nesting=nesting, embed=embed_model)


    def add_model(
            self,
            model,
            data_statistics=True,
            nesting=True,
            utility_functions=True,
            on_error='pass',
            embed=True,
    ):

        try:
            self.add_content_tab(model.parameter_summary('df'), sheetname="Parameters", heading="Parameters" )
        except:
            if on_error == 'raise':
                raise

        if data_statistics:
            from .statistics import statistics_for_dataframe
            if model.dataframes is not None:
                if model.dataframes.data_co is not None and len(model.dataframes.data_co.columns):
                    try:
                        self.add_content_tab(
                            statistics_for_dataframe(model.dataframes.data_co),
                            sheetname="CO Data",
                            heading="CO Data",
                        )
                    except:
                        if on_error=='raise':
                            raise
                if model.dataframes.data_ca is not None and len(model.dataframes.data_ca.columns):
                    try:
                        self.add_content_tab(
                            statistics_for_dataframe(model.dataframes.data_ca),
                            sheetname="CA Data",
                            heading="CA Data",
                        )
                    except:
                        if on_error=='raise':
                            raise
                if model.dataframes.data_ce is not None and len(model.dataframes.data_ce.columns):
                    try:
                        self.add_content_tab(
                            statistics_for_dataframe(model.dataframes.data_ce),
                            sheetname="CE Data",
                            heading="CE Data",
                        )
                    except:
                        if on_error=='raise':
                            raise
                if model.dataframes.data_ch is not None and model.dataframes.data_av is not None:
                    try:
                        self.add_content_tab(
                            model.dataframes.choice_avail_summary(graph=model.graph),
                            sheetname="Choice",
                            heading="Choices",
                        )
                    except:
                        if on_error=='raise':
                            raise

        if utility_functions:
            try:
                self.add_content_tab(model._utility_functions_as_frame(), sheetname="Utility", heading="Utility Functions")
                self.sheets["Utility"].set_column('B:B', None, None, {'hidden': 1})
            except:
                if on_error == 'raise':
                    raise

        if nesting and not model.is_mnl():
            try:
                self.add_content_tab(model.graph.__xml__(output='png'), sheetname="Nesting", heading="Nesting Tree")
            except:
                if on_error == 'raise':
                    raise
            try:
                self.add_content_tab(model.graph_descrip('nodes'), sheetname="Nesting", heading="Nesting Node List")
            except:
                if on_error == 'raise':
                    raise

        if embed:
            self.add_metadata('_self_', model)

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

    def _add_toc_entry(self, heading, target_sheet, target_row):
        self.tocsheet.write_url(self.sheet_startrow['Contents'], 0, f"internal:'{target_sheet}'!A{target_row+1}", string=heading)
        self.sheet_startrow['Contents'] += 1

    def add_content_tab(self, content, sheetname, heading=None, startrow=None, blurb=None):

        if startrow is None:
            startrow = self.sheet_startrow.get(sheetname, 0)
        worksheet = self.add_worksheet(sheetname)

        content_in = content
        success = False

        logger.debug("writing to %s", sheetname)

        if isinstance(content, (pd.DataFrame, Styler)):

            if isinstance(content, Styler):
                content_, content_data = content, content.data
            else:
                content_, content_data = content, content

            if heading is not None:
                worksheet.write_url(startrow, 0, 'internal:Contents!A1', self.toc_link_fmt, string='<< Back to Table of Contents', )
                startrow += 1
                h = self.TAB(heading)
                worksheet.write(startrow, 0, h, self.head_fmt)
                self._add_toc_entry(h, sheetname, startrow)
                startrow += 1
                logger.debug(" after heading row is %d", startrow)
            content_.to_excel(self, sheet_name=worksheet.name, startrow=startrow)
            startrow += len(content_data) + content.columns.nlevels
            if content_data.index.name:
                startrow += 1
            startrow += 2 # gap
            logger.debug(" after table row is %d", startrow)
            success = True

            if sheetname not in self._col_widths:
                self._col_widths[sheetname] = {}
            if content_data.index.nlevels == 1:
                current_width = self._col_widths[sheetname].get(0, 8)
                new_width = max(current_width, max(len(str(i)) for i in content_data.index))
                self._col_widths[sheetname][0] = new_width
                worksheet.set_column(0, 0, new_width)
            else:
                for n in range(content_data.index.nlevels):
                    current_width = self._col_widths[sheetname].get(n, 8)
                    new_width = max(current_width, max(len(str(i)) for i in content_data.index.levels[n]))
                    self._col_widths[sheetname][n] = new_width
                    worksheet.set_column(n,n,new_width)

        # Render matplotlib.Figure into an Elem
        if not success and 'matplotlib' in str(type(content)):
            if isinstance(content, matplotlib.figure.Figure):
                try:
                    content = make_png(content, dpi='figure', compress=True, output='Elem', facecolor='w')
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
                    dpi = _v.attrib['dpi']
                except:
                    dpi = None

                try:
                    _v = _v.attrib['src']
                except:
                    pass
                else:
                    if isinstance(_v, str) and _v[:22] == 'data:image/png;base64,':
                        fp, img_size, dpi = make_png(
                            base64.standard_b64decode(_v[22:]), dpi=dpi, output='bytesio',
                            return_size=True, return_dpi=True,
                        )
                        if heading is not None:
                            worksheet.write_url(startrow, 0, 'internal:Contents!A1', self.toc_link_fmt,
                                                string='<< Back to Table of Contents', )
                            startrow += 1
                            h = self.FIG(heading)
                            worksheet.write(startrow, 0, h, self.head_fmt)
                            self._add_toc_entry(h, sheetname, startrow)
                            startrow += 1
                            logger.debug(" after heading row is %d", startrow)
                        worksheet.insert_image(startrow, 0, f'image-{sheetname}.png', {'image_data': fp})
                        startrow += int(np.ceil(img_size[1] / dpi[1] * 72 / 15)) + 3
                        logger.debug(" after image row is %d", startrow)
                        success = True

        if not success:
            import warnings
            warnings.warn(f'content not written to sheet {sheetname}, type {type(content_in)}')
        else:
            if blurb is not None:
                import textwrap
                blurb = textwrap.wrap(blurb)
                for line in blurb:
                    worksheet.write(startrow, 0, line)
                    startrow += 1
                logger.debug(" after blurb row is %d", startrow)

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

def _make_excel_writer(model, filename, save_now=True, **kwargs):
    """
    Write the model to an Excel file.

    Parameters
    ----------
    model : larch.Model
    filename : str
    save_now : bool, default True
        Save the model immediately.  Set to False if you want to
        write additional figures or tables to the file before saving.
    **kwargs
        Other keyword arguments are passed to the `ExcelWriter`
        constructor.

    Returns
    -------
    larch.util.excel.ExcelWriter
    """
    xl = ExcelWriter(filename, engine='xlsxwriter_larch', model=model, **kwargs)
    if save_now:
        xl.save()
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

