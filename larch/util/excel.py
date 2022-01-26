
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from xmle import Elem
import matplotlib.figure, matplotlib.axes
import io
import time
import base64
from .png import make_png
from .. import __version__
import logging
from ..model.model_group import ModelGroup

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
        options = kwargs.pop('workbook_options', {})
        if 'nan_inf_to_errors' not in options:
            options['nan_inf_to_errors'] = True
        super().__init__(*args, engine=_engine, options=options, engine_kwargs=kwargs)
        self.book.set_size(1600, 1200)
        self.head_fmt = self.book.add_format({'bold': True, 'font_size':14})
        self.ital_fmt = self.book.add_format({'italic': True, 'font_size':12})
        self.toc_link_fmt = self.book.add_format({'font_size':8})

        self.fixed_precision = {
            2: "_-0.00;-0.00"
        }

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
        if model is not None:
            self.tocsheet.write(0, 0, model.title, self.head_fmt)
        self.tocsheet.write(2, 0, f'   🌳 Larch {__version__}')
        self.tocsheet.write(3, 0, time.strftime("     %A %d %B %Y, %I:%M:%S %p %Z"))
        self.tocsheet.write(5, 0, 'Table of Contents', self.head_fmt)
        self.sheet_startrow['Contents'] = 6

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

        ps_data_index_nlevels = 1
        try:
            ps = model.parameter_summary('df')
            ps_data_index_nlevels = ps.data.index.nlevels
            if 'Constrained' in ps.data.columns:
                ps.data['Constrained'] = ps.data['Constrained'].str.replace("<br/>",",\n")
            self.add_content_tab(ps, sheetname="Parameters", heading="Parameters" )
            if 'Constrained' in ps.data.columns:
                n = ps.data.index.nlevels + ps.data.columns.get_loc("Constrained")
                if "Parameters" not in self._col_widths:
                    self._col_widths["Parameters"] = {}
                current_width = self._col_widths["Parameters"].get(n, 8)
                new_width = max(current_width, 50)
                self._col_widths["Parameters"][n] = new_width
                wrap = self.book.add_format()
                wrap.set_text_wrap()
                self.sheets["Parameters"].set_column(n, n, width=new_width, cell_format=wrap)
        except:
            if on_error == 'raise':
                raise

        try:
            startrow = self.sheet_startrow.get("Parameters", 0)
            worksheet = self.sheets["Parameters"]
            worksheet.write_url(startrow, 0, 'internal:Contents!A1', self.toc_link_fmt, string='<< Back to Table of Contents', )
            startrow += 1
            h = self.TAB("Estimation Statistics")
            worksheet.write(startrow, 0, h, self.head_fmt)
            self._add_toc_entry(h, "Parameters", startrow)
            startrow += 1
            logger.debug(" after heading row is %d", startrow)
            startrow = _estimation_statistics_excel(
                model,
                self,
                "Parameters",
                start_row=startrow,
                buffer_cols=ps_data_index_nlevels-1,
            )
        except:
            if on_error == 'raise':
                raise

        if data_statistics and not isinstance(model, ModelGroup):
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
                            model.dataframes.choice_avail_summary(
                                graph=model.graph,
                                availability_co_vars=model.availability_co_vars,
                            ),
                            sheetname="Choice",
                            heading="Choices",
                        )
                    except:
                        if on_error=='raise':
                            raise

        if utility_functions:
            try:
                if isinstance(model, ModelGroup):
                    for m in model:
                        self.add_content_tab(
                            m._utility_functions_as_frame(),
                            sheetname="Utility",
                            heading=f"Utility Functions {m.title}",
                        )
                else:
                    self.add_content_tab(model._utility_functions_as_frame(), sheetname="Utility", heading="Utility Functions")
                self.sheets["Utility"].set_column('B:B', None, None, {'hidden': 1})
            except:
                if on_error == 'raise':
                    raise

        if nesting and not isinstance(model, ModelGroup) and not model.is_mnl():
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

        elif nesting and isinstance(model, ModelGroup):
            for m in model:
                if not m.is_mnl():
                    try:
                        self.add_content_tab(m.graph.__xml__(output='png'), sheetname="Nesting", heading="Nesting Tree")
                    except:
                        if on_error == 'raise':
                            raise
                    try:
                        self.add_content_tab(m.graph_descrip('nodes'), sheetname="Nesting", heading="Nesting Node List")
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

    def add_content_tab(self, content, sheetname, heading=None, startrow=None, blurb=None, to_excel=None):

        if startrow is None:
            startrow = self.sheet_startrow.get(sheetname, 0)
        worksheet = self.add_worksheet(sheetname)

        content_in = content
        success = False
        if to_excel is None:
            to_excel = {}

        logger.debug("writing to %s", sheetname)

        if isinstance(content, (pd.DataFrame, Styler)):
            logger.debug("writing as DataFrame (%s)", type(content))

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
            content_.to_excel(self, sheet_name=worksheet.name, startrow=startrow, **to_excel)
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
            logger.debug("writing as matplotlib (%s)", type(content))
            if isinstance(content, matplotlib.axes.Axes):
                content = content.get_figure()
            if isinstance(content, matplotlib.figure.Figure):
                try:
                    content = make_png(content, dpi='figure', compress=True, output='Elem', facecolor='w')
                except:
                    pass

        # Extract PNG data from Elem if found there
        if not success and isinstance(content, Elem):
            logger.debug("writing as Elem (%s)", type(content))
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

        if not success and isinstance(content, dict):
            logger.debug("writing as dict (%s)", type(content))
            max_row = 0
            max_col = 0
            row = -1
            prev_col = 0
            cellcontent = []
            from collections.abc import Iterable
            def unpacker(i, indent=0):
                nonlocal row, prev_col, max_row, max_col, cellcontent
                if isinstance(i, dict):
                    for k, v in i.items():
                        unpacker(k, indent)
                        unpacker(v, indent + 1)
                elif isinstance(i, pd.Series):
                    for k, v in dict(i).items():
                        unpacker(k, indent)
                        unpacker(v, indent + 1)
                elif not isinstance(i, str) and isinstance(i, Iterable):
                    for k in i:
                        unpacker(k, indent)
                else:
                    if prev_col >= indent:
                        row = row + 1
                    cellcontent.append((indent, row, i))
                    max_col = max(max_col, indent)
                    max_row = max(max_row, row)
                    prev_col = indent
            unpacker(content)
            table_cache = pd.DataFrame(data="", index=pd.RangeIndex(max_row + 1), columns=pd.RangeIndex(max_col + 1))
            for c in cellcontent:
                table_cache.iloc[c[1], c[0]] = c[2]
            self.add_content_tab(
                table_cache,
                sheetname=sheetname,
                heading=heading,
                startrow=startrow,
                blurb=blurb,
                to_excel=dict(
                    index=False,
                    header=False,
                ),
            )
            if sheetname not in self._col_widths:
                self._col_widths[sheetname] = {}
            for n in range(max_col):
                current_width = self._col_widths[sheetname].get(n, 8)
                new_width = max(current_width, 16)
                self._col_widths[sheetname][n] = new_width
                worksheet.set_column(n,n,new_width)
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
        if self.path is not None:
            if makedirs:
                import os
                dirname = os.path.dirname(self.path)
                if dirname:
                    os.makedirs(dirname, exist_ok=True)

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
    model : larch.Model or larch.ModelGroup
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
    if xlsxwriter is None:
        raise RuntimeError('xlsxwriter is not installed')
    xl = ExcelWriter(filename, engine='xlsxwriter_larch', model=model, **kwargs)
    if save_now:
        xl.save()
    return xl

from .. import Model
Model.to_xlsx = _make_excel_writer

def load_metadata(xlsx_filename, key=None):
    import pickle, base64, pandas
    raw = pandas.read_excel(xlsx_filename, sheet_name='_metadata_', index_col=0, engine='openpyxl')
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


def _estimation_statistics_excel(
        model,
        xlsxwriter,
        sheetname,
        start_row=0,
        start_col=0,
        buffer_cols=0,
        compute_loglike_null=True,
):
    """
    Write a tabular summary of estimation statistics to excel.

    This will generate a small table of estimation statistics,
    containing:

    *	Log Likelihood at Convergence
    *	Log Likelihood at Null Parameters (if known)
    *	Log Likelihood with No Model (if known)
    *	Log Likelihood at Constants Only (if known)

    Additionally, for each included reference value (i.e.
    everything except log likelihood at convergence) the
    rho squared with respect to that value is also given.

    Each statistic is reported in aggregate, as well as
    per case.

    Parameters
    ----------
    xlsxwriter : ExcelWriter
    sheetname : str
    start_row, start_col : int
        Zero-based index of upper left cell
    buffer_cols : int
        Number of extra columns between statistic label and
        values.
    compute_loglike_null : bool, default True
        If the log likelihood at null values has not already
        been computed (i.e., if it is not cached) then compute
        it, cache its value, and include it in the output.

    """
    try:
        from ..exceptions import MissingDataError, BHHHSimpleStepFailure

        if start_row is None:
            start_row = xlsxwriter.sheet_startrow.get(sheetname, 0)
        worksheet = xlsxwriter.add_worksheet(sheetname)

        row = start_row

        fixed_2 = xlsxwriter.book.add_format({'num_format': '#,##0.00'})
        fixed_4 = xlsxwriter.book.add_format({'num_format': '0.0000'})
        comma_0 = xlsxwriter.book.add_format({'num_format': '#,##0'})
        bold = xlsxwriter.book.add_format({'bold': True})
        bold_centered = xlsxwriter.book.add_format({'bold': True})

        fixed_2.set_align('center')
        fixed_4.set_align('center')
        comma_0.set_align('center')
        bold_centered.set_align('center')
        bold.set_border(1)
        bold_centered.set_border(1)

        datum_col = start_col + buffer_cols

        def catname(j):
            nonlocal row, start_col, buffer_cols
            if buffer_cols:
                worksheet.merge_range(row, start_col, row, start_col + buffer_cols, j, bold)
            else:
                worksheet.write(row, start_col, j, bold)

        catname('Statistic')
        worksheet.write(row, datum_col + 1, 'Aggregate', bold_centered)
        worksheet.write(row, datum_col + 2, 'Per Case', bold_centered)
        row += 1

        try:
            ncases = model.n_cases
        except MissingDataError:
            ncases = None

        catname('Number of Cases')
        if ncases:
            worksheet.merge_range(row, datum_col + 1, row, datum_col + 2, ncases, cell_format=comma_0)
        else:
            worksheet.merge_range(row, datum_col + 1, row, datum_col + 2, "not available")
        row += 1

        mostrecent = model._most_recent_estimation_result
        if mostrecent is not None:
            catname('Log Likelihood at Convergence')
            worksheet.write(row, datum_col + 1, mostrecent.loglike, fixed_2)  # "{:.2f}".format(mostrecent.loglike)
            if ncases:
                worksheet.write(row, datum_col + 2, mostrecent.loglike / ncases,
                                fixed_4)  # "{:.2f}".format(mostrecent.loglike/ ncases)
            else:
                worksheet.write(row, datum_col + 2, "na")
            row += 1

        ll_z = model._cached_loglike_null
        if ll_z == 0:
            if compute_loglike_null:
                try:
                    ll_z = model.loglike_null()
                except MissingDataError:
                    pass
                else:
                    model.loglike()
            else:
                ll_z = 0
        if ll_z != 0:
            catname('Log Likelihood at Null Parameters')
            worksheet.write(row, datum_col + 1, ll_z, fixed_2)  # "{:.2f}".format(ll_z)
            if ncases:
                worksheet.write(row, datum_col + 2, ll_z / ncases, fixed_4)  # "{:.2f}".format(ll_z/ ncases)
            else:
                worksheet.write(row, datum_col + 2, "na")
            if mostrecent is not None:
                row += 1
                catname('Rho Squared w.r.t. Null Parameters')
                rsz = 1.0 - (mostrecent.loglike / ll_z)
                worksheet.merge_range(row, datum_col + 1, row, datum_col + 2, rsz,
                                      cell_format=fixed_4)  # "{:.3f}".format(rsz)
            row += 1

        ll_nil = model._cached_loglike_nil
        if ll_nil != 0:
            catname('Log Likelihood with No Model')
            worksheet.write(row, datum_col + 1, ll_nil, fixed_2)  # "{:.2f}".format(ll_nil)
            if ncases:
                worksheet.write(row, datum_col + 2, ll_nil / ncases, fixed_4)  # "{:.2f}".format(ll_nil/ ncases)
            else:
                worksheet.write(row, datum_col + 2, "na")
            if mostrecent is not None:
                row += 1
                catname('Rho Squared w.r.t. No Model')
                rsz = 1.0 - (mostrecent.loglike / ll_nil)
                worksheet.merge_range(row, datum_col + 1, row, datum_col + 2, rsz,
                                      cell_format=fixed_4)  # "{:.3f}".format(rsz)
            row += 1

        ll_c = model._cached_loglike_constants_only
        if ll_c != 0:
            catname('Log Likelihood at Constants Only')
            worksheet.write(row, datum_col + 1, ll_c, fixed_2)  # "{:.2f}".format(ll_c)
            if ncases:
                worksheet.write(row, datum_col + 2, ll_c / ncases, fixed_4)  # "{:.2f}".format(ll_c/ ncases)
            else:
                worksheet.write(row, datum_col + 2, "na")
            if mostrecent is not None:
                row += 1
                catname('Rho Squared w.r.t. Constants Only')
                rsc = 1.0 - (mostrecent.loglike / ll_c)
                worksheet.merge_range(row, datum_col + 1, row, datum_col + 2, rsc,
                                      cell_format=fixed_4)  # "{:.3f}".format(rsc)
            row += 1

        if mostrecent is not None:
            if 'message' in mostrecent:
                catname('Optimization Message')
                worksheet.write(row, datum_col + 1, mostrecent.message)
                row += 1

        if sheetname not in xlsxwriter._col_widths:
            xlsxwriter._col_widths[sheetname] = {}
        current_width = xlsxwriter._col_widths[sheetname].get(start_col, 8)
        proposed_width = 28
        if buffer_cols:
            for b in range(buffer_cols):
                proposed_width -= xlsxwriter._col_widths[sheetname].get(start_col + 1 + b, 8)
        new_width = max(current_width, proposed_width)
        xlsxwriter._col_widths[sheetname][start_col] = new_width
        worksheet.set_column(start_col, start_col, new_width)

        row += 2  # gap
        xlsxwriter.sheet_startrow[worksheet.name] = row
        return row
    except:
        logger.exception("error in _estimation_statistics_excel")
        raise