#
# Adapted from the base pandas Styler.background_gradient
# https://github.com/pandas-dev/pandas/blob/v1.0.2/pandas/io/formats/style.py
#
# BSD 3-Clause License
#
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import numpy as np
import pandas as pd
from pandas.io.formats.style import _mpl, Styler
from typing import Optional
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.indexes.api import Index
from pandas.api.types import is_dict_like, is_list_like


# the public IndexSlicerMaker
class _IndexSlice:
	"""
	Create an object to more easily perform multi-index slicing.

	See Also
	--------
	MultiIndex.remove_unused_levels : New MultiIndex with no unused levels.

	Notes
	-----
	See :ref:`Defined Levels <advanced.shown_levels>`
	for further info on slicing a MultiIndex.

	Examples
	--------
	>>> midx = pd.MultiIndex.from_product([['A0','A1'], ['B0','B1','B2','B3']])
	>>> columns = ['foo', 'bar']
	>>> dfmi = pd.DataFrame(np.arange(16).reshape((len(midx), len(columns))),
							index=midx, columns=columns)

	Using the default slice command:

	>>> dfmi.loc[(slice(None), slice('B0', 'B1')), :]
			   foo  bar
		A0 B0    0    1
		   B1    2    3
		A1 B0    8    9
		   B1   10   11

	Using the IndexSlice class for a more intuitive command:

	>>> idx = pd.IndexSlice
	>>> dfmi.loc[idx[:, 'B0':'B1'], :]
			   foo  bar
		A0 B0    0    1
		   B1    2    3
		A1 B0    8    9
		   B1   10   11
	"""

	def __getitem__(self, arg):
		return arg


IndexSlice = _IndexSlice()


def _non_reducing_slice(slice_):
	"""
	Ensure that a slice doesn't reduce to a Series or Scalar.

	Any user-passed `subset` should have this called on it
	to make sure we're always working with DataFrames.
	"""
	# default to column slice, like DataFrame
	# ['A', 'B'] -> IndexSlices[:, ['A', 'B']]
	kinds = (ABCSeries, np.ndarray, Index, list, str)
	if isinstance(slice_, kinds):
		slice_ = IndexSlice[:, slice_]

	def pred(part) -> bool:
		"""
		Returns
		-------
		bool
			True if slice does *not* reduce,
			False if `part` is a tuple.
		"""
		# true when slice does *not* reduce, False when part is a tuple,
		# i.e. MultiIndex slice
		return (isinstance(part, slice) or is_list_like(part)) and not isinstance(
			part, tuple
		)

	if not is_list_like(slice_):
		if not isinstance(slice_, slice):
			# a 1-d slice, like df.loc[1]
			slice_ = [[slice_]]
		else:
			# slice(a, b, c)
			slice_ = [slice_]  # to tuplize later
	else:
		slice_ = [part if pred(part) else [part] for part in slice_]
	return tuple(slice_)


def _maybe_numeric_slice(df, slice_, include_bool=False):
	"""
	Want nice defaults for background_gradient that don't break
	with non-numeric data. But if slice_ is passed go with that.
	"""
	if slice_ is None:
		dtypes = [np.number]
		if include_bool:
			dtypes.append(bool)
		slice_ = IndexSlice[:, df.select_dtypes(include=dtypes).columns]
	return slice_

class OverspecView(Styler):

	def __init__(self, data, *args, cmap='YlOrRd', high=0.3, **kwargs):
		if isinstance(data, (tuple,list)):
			data_ = pd.concat([
				pd.DataFrame(
					eigvec,
					index=eigpar,
					columns=[f"({n + 1}) {eigval:.4g}"]
				)
				for n, (eigval, eigpar, eigvec) in enumerate(data)
			], axis=1)
		elif data is None:
			data_ = pd.DataFrame()
		else:
			data_ = data
		super().__init__( data_, *args, **kwargs )
		self.absolute_background_gradient(cmap=cmap, high=high)

	def absolute_background_gradient(
			self,
			cmap="YlOrRd",
			low=0,
			high=0,
			axis=0,
			subset=None,
			text_color_threshold=0.408,
			vmin: Optional[float] = 0,
			vmax: Optional[float] = None,
	):
		"""
		Color the background in a gradient style,
		based on the magnitude (absolute value).
		The background color is determined according
		to the data in each column (optionally row).
		Requires matplotlib.


		Parameters
		----------
		cmap : str or colormap
			Matplotlib colormap.
		low : float
			Compress the range by the low.
		high : float
			Compress the range by the high.
		axis : {0 or 'index', 1 or 'columns', None}, default 0
			Apply to each column (``axis=0`` or ``'index'``), to each row
			(``axis=1`` or ``'columns'``), or to the entire DataFrame at once
			with ``axis=None``.
		subset : IndexSlice
			A valid slice for ``data`` to limit the style application to.
		text_color_threshold : float or int
			Luminance threshold for determining text color. Facilitates text
			visibility across varying background colors. From 0 to 1.
			0 = all text is dark colored, 1 = all text is light colored.
		vmin : float, default 0
			Minimum data value that corresponds to colormap minimum value.
			When None, the minimum value of the data will be used.
		vmax : float, optional
			Maximum data value that corresponds to colormap maximum value.
			When None (default): the maximum value of the data will be used.
		Returns
		-------
		self : Styler
		Raises
		------
		ValueError
			If ``text_color_threshold`` is not a value from 0 to 1.
		Notes
		-----
		Set ``text_color_threshold`` or tune ``low`` and ``high`` to keep the
		text legible by not using the entire range of the color map. The range
		of the data is extended by ``low * (x.max() - x.min())`` and ``high *
		(x.max() - x.min())`` before normalizing.
		"""
		subset = _maybe_numeric_slice(self.data, subset)
		subset = _non_reducing_slice(subset)
		self.apply(
			self._absolute_background_gradient,
			cmap=cmap,
			subset=subset,
			axis=axis,
			low=low,
			high=high,
			text_color_threshold=text_color_threshold,
			vmin=vmin,
			vmax=vmax,
		)
		return self

	@staticmethod
	def _absolute_background_gradient(
			s,
			cmap="YlOrRd",
			low=0,
			high=0,
			text_color_threshold=0.408,
			vmin: Optional[float] = 0,
			vmax: Optional[float] = None,
	):
		"""
		Color background in a range according to the data.
		"""
		if (
				not isinstance(text_color_threshold, (float, int))
				or not 0 <= text_color_threshold <= 1
		):
			msg = "`text_color_threshold` must be a value from 0 to 1."
			raise ValueError(msg)

		with _mpl(Styler.background_gradient) as (plt, colors):
			smin = np.nanmin(np.absolute(s.to_numpy())) if vmin is None else vmin
			smax = np.nanmax(np.absolute(s.to_numpy())) if vmax is None else vmax
			rng = smax - smin
			# extend lower / upper bounds, compresses color range
			norm = colors.Normalize(smin - (rng * low), smax + (rng * high))
			# matplotlib colors.Normalize modifies inplace?
			# https://github.com/matplotlib/matplotlib/issues/5427
			rgbas = plt.cm.get_cmap(cmap)(norm(np.absolute(s.to_numpy(dtype=float))))

			def relative_luminance(rgba):
				"""
				Calculate relative luminance of a color.
				The calculation adheres to the W3C standards
				(https://www.w3.org/WAI/GL/wiki/Relative_luminance)
				Parameters
				----------
				color : rgb or rgba tuple
				Returns
				-------
				float
					The relative luminance as a value from 0 to 1
				"""
				r, g, b = (
					x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055 ** 2.4)
					for x in rgba[:3]
				)
				return 0.2126 * r + 0.7152 * g + 0.0722 * b

			def css(rgba):
				dark = relative_luminance(rgba) < text_color_threshold
				text_color = "#f1f1f1" if dark else "#000000"
				return f"background-color: {colors.rgb2hex(rgba)};color: {text_color};"

			if s.ndim == 1:
				return [css(rgba) for rgba in rgbas]
			else:
				return pd.DataFrame(
					[[css(rgba) for rgba in row] for row in rgbas],
					index=s.index,
					columns=s.columns,
				)