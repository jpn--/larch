
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

# partly based on ReflectionBoundedKDE from
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


def weighted_sample_std(values, weights, ddof=1.0):
    """
    Weighted sample standard deviation.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    variance = variance * sum(weights) / (sum(weights) - ddof)
    return np.sqrt(variance)


class BoundedKDE:

    def __init__(self, x, weights=None, bw_method='scott', lb='min', ub='max', kernel='gaussian'):
        if lb is None:
            self.lower_bound = None
        else:
            self.lower_bound = lb if lb != 'min' else np.nanmin(x)
        if ub is None:
            self.upper_bound = None
        else:
            self.upper_bound = ub if ub != 'max' else np.nanmax(x)
        if x.ndim > 1:
            x = x.ravel()
            if weights is not None:
                weights = weights.ravel()
        if weights is not None:
            x = x[weights>0]
            weights = weights[weights>0]
        if kernel is None:
            if x.size > 1000:
                kernel = 'tophat'
            else:
                kernel = 'gaussian'
        if bw_method == 'scott':
            if weights is None:
                bw = x.std(ddof=1) * np.power(x.size, -1.0 / 5)
            else:
                bw = weighted_sample_std(x, weights, ddof=1.0) * np.power(weights.sum(), -1.0 / 5)
        elif getattr(bw_method, 'bandwidth', None) is not None:
            bw = getattr(bw_method, 'bandwidth')
        elif np.isscalar(bw_method) and not isinstance(bw_method, str):
            bw = bw_method
        else:
            raise NotImplementedError(f"{bw_method=}")
        self.bandwidth = bw
        self.kernel = kernel
        self.kde = KernelDensity(bandwidth=bw, kernel=self.kernel)
        self.kde.fit(x[:, np.newaxis], sample_weight=weights)

    def evaluate(self, x):
        if x.ndim > 1:
            x = x.ravel()
        pdf = np.exp(self.kde.score_samples(x[:, np.newaxis]))
        if self.lower_bound is not None:
            pdf += np.exp(self.kde.score_samples((2 * self.lower_bound - x)[:, np.newaxis]))
        if self.upper_bound is not None:
            pdf += np.exp(self.kde.score_samples((2 * self.upper_bound - x)[:, np.newaxis]))
        return pdf

    def __call__(self, x):
        if x.ndim > 1:
            x = x.ravel()
        results = self.evaluate(x)
        if self.lower_bound is not None:
            results[x < self.lower_bound] = 0.0
        if self.upper_bound is not None:
            results[x > self.upper_bound] = 0.0
        return results


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
