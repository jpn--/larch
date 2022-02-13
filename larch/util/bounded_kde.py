

# based on ReflectionBoundedKDE from
# https://git.ligo.org/lscsoft/pesummary/-/blob/master/pesummary/core/plots/bounded_1d_kde.py
#
# MIT License
# Copyright (c) 2018-2021 Charlie Hoy charlie.hoy@ligo.org
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.stats import gaussian_kde

class bounded_gaussian_kde(gaussian_kde):

    def __init__(self, dataset, bw_method=None, weights=None, lb='min', ub='max'):
        super().__init__(
            dataset, bw_method=bw_method, weights=weights,
        )
        self.lower_bound = lb if lb != 'min' else np.nanmin(dataset)
        self.upper_bound = ub if ub != 'max' else np.nanmax(dataset)

    def evaluate(self, pts):
        """
        Return an estimate of the density evaluated at the given points
        """
        x = pts.T
        pdf = super().evaluate(pts.T)
        if self.lower_bound is not None:
            pdf += super().evaluate(2 * self.lower_bound - x)
        if self.upper_bound is not None:
            pdf += super().evaluate(2 * self.upper_bound - x)
        return pdf

    def __call__(self, pts):
        pts = np.atleast_1d(pts)
        results = self.evaluate(pts)
        if self.lower_bound is not None:
            results[pts < self.lower_bound] = 0.0
        if self.upper_bound is not None:
            results[pts > self.upper_bound] = 0.0
        return results
