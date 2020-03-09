
import numpy as np
from scipy.optimize import LinearConstraint
from abc import ABC, abstractmethod

class ParametricConstraint(ABC):

    @abstractmethod
    def fun(self, x):
        raise NotImplementedError("abstract base class, use a derived class instead")

    @abstractmethod
    def jac(self, x):
        raise NotImplementedError("abstract base class, use a derived class instead")

    def as_constraint_dicts(self):
        return [dict(type='ineq', fun=self.fun, jac=self.jac),]


class RatioBound(ParametricConstraint):

    def __init__(self, model, p_num, p_den, min_ratio=None, max_ratio=None, scale=10):
        self.p_num = p_num
        self.p_den = p_den
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.scale = 1
        self.link_model(model, scale)

    def link_model(self, model, scale=None):
        self.i_num = model.pf.index.get_loc(self.p_num)
        self.i_den = model.pf.index.get_loc(self.p_den)
        self.len = len(model.pf)
        if scale is not None:
            self.scale = scale
        if self.min_ratio is not None:
            scaling = max(self.min_ratio, 1/self.min_ratio) * self.scale
            if model.pf.loc[self.p_den, 'minimum'] == 0:
                # positive denominator
                self.cmin_num = 1 * scaling
                self.cmin_den = -self.min_ratio * scaling
            elif model.pf.loc[self.p_den, 'maximum'] == 0:
                # negative denominator
                self.cmin_num = -1 * scaling
                self.cmin_den = self.min_ratio * scaling
            else:
                raise ValueError('denominator must be bounded at zero')
        else:
            self.cmin_num = 0
            self.cmin_den = 0
        if self.max_ratio is not None:
            scaling = max(self.max_ratio, 1/self.max_ratio) * self.scale
            if model.pf.loc[self.p_den, 'minimum'] == 0:
                # positive denominator
                self.cmax_num = -1 * scaling
                self.cmax_den = self.max_ratio * scaling
            elif model.pf.loc[self.p_den, 'maximum'] == 0:
                # negative denominator
                self.cmax_num = 1 * scaling
                self.cmax_den = -self.max_ratio * scaling
            else:
                raise ValueError('denominator must be bounded at zero')
        else:
            self.cmax_num = 0
            self.cmax_den = 0

    def _min_fun(self, x):
        return x[self.i_num] * self.cmin_num + x[self.i_den] * self.cmin_den

    def _max_fun(self, x):
        return x[self.i_num] * self.cmax_num + x[self.i_den] * self.cmax_den

    def fun(self, x):
        return min(
            self._min_fun(x),
            self._max_fun(x),
        )

    def jac(self, x):
        j = np.zeros_like(x)
        if self._min_fun(x) < self._max_fun(x):
            j[self.i_num] = self.cmin_num
            j[self.i_den] = self.cmin_den
        else:
            j[self.i_num] = self.cmax_num
            j[self.i_den] = self.cmax_den
        return j

    def as_linear_constraints(self):
        a = np.zeros([2,self.len], dtype='float64')
        a[0,self.i_num] = self.cmin_num
        a[0,self.i_den] = self.cmin_den
        a[1,self.i_num] = self.cmax_num
        a[1,self.i_den] = self.cmax_den
        return [LinearConstraint(a, 0, np.inf)]

class OrderingBound(ParametricConstraint):

    def __init__(self, model, p_less, p_more, scale=10):
        self.p_less = p_less
        self.p_more = p_more
        self.len = len(model.pf)
        self.scale = 1
        self.link_model(model, scale)

    def link_model(self, model, scale=None):
        self.i_less = model.pf.index.get_loc(self.p_less)
        self.i_more = model.pf.index.get_loc(self.p_more)
        self.len = len(model.pf)
        if scale is not None:
            self.scale = scale

    def fun(self, x):
        return (x[self.i_more] - x[self.i_less])*self.scale

    def jac(self, x):
        j = np.zeros_like(x)
        j[self.i_more] = self.scale
        j[self.i_less] = -self.scale
        return j

    def as_linear_constraints(self):
        a = np.zeros([1,self.len], dtype='float64')
        a[0,self.i_more] = self.scale
        a[0,self.i_less] = -self.scale
        return [LinearConstraint(a, 0, np.inf)]
