import numpy as np
import pandas as pd
from numba import guvectorize, njit
from numba import int8 as i8
from numba import int32 as i32
from numba import float32 as f32
from numba import float64 as f64
from numba import boolean
from ..model import Model as _BaseModel
from ..exceptions import MissingDataError
from ..util import dictx

import warnings
warnings.warn(
    "\n\n"
    "*** larch.numba is experimental, use at your own risk ***\n"
    " on first import loading this package will compile optimized\n"
    " binaries for your machine, which may take several seconds\n"
)

@njit(error_model='numpy', fastmath=True, cache=True)
def utility_from_data_co(
        model_utility_co_alt,          # int input shape=[n_co_features]
        model_utility_co_param_scale,  # float input shape=[n_co_features]
        model_utility_co_param,        # int input shape=[n_co_features]
        model_utility_co_data,         # int input shape=[n_co_features]
        parameter_arr,                 # float input shape=[n_params]
        holdfast_arr,                  # float input shape=[n_params]
        array_av,                      # int8 input shape=[n_alts]
        data_co,                       # float input shape=[n_co_vars]
        utility_elem,                  # float output shape=[n_alts]
        dutility_elem,                 # float output shape=[n_alts, n_params]
):
    for i in range(model_utility_co_alt.shape[0]):
        altindex = model_utility_co_alt[i]
        param_value = parameter_arr[model_utility_co_param[i]]
        param_holdfast = holdfast_arr[model_utility_co_param[i]]
        if array_av[altindex]:
            if model_utility_co_data[i] == -1:
                utility_elem[altindex] += param_value * model_utility_co_param_scale[i]
                if not param_holdfast:
                    dutility_elem[altindex, model_utility_co_param[i]] += model_utility_co_param_scale[i]
            else:
                _temp = data_co[model_utility_co_data[i]] * model_utility_co_param_scale[i]
                utility_elem[altindex] += _temp * param_value
                if not param_holdfast:
                    dutility_elem[altindex, model_utility_co_param[i]] += _temp



@njit(error_model='numpy', fastmath=True, cache=True)
def utility_from_data_ca(
        model_utility_ca_param_scale,   # int input shape=[n_u_ca_features]
        model_utility_ca_param,         # int input shape=[n_u_ca_features]
        model_utility_ca_data,          # int input shape=[n_u_ca_features]
        parameter_arr,                  # float input shape=[n_params]
        holdfast_arr,                   # float input shape=[n_params]
        array_av,                       # int8 input shape=[n_alts]
        array_ca,                       # float input shape=[n_alts, n_ca_vars]
        utility_elem,                   # float output shape=[n_alts]
        dutility_elem,                  # float output shape=[n_alts, n_params]
):
    n_alts = array_ca.shape[0]
    for j in range(n_alts):

        # if self._array_ce_reversemap is not None:
        #     if c >= self._array_ce_reversemap.shape[0] or j >= self._array_ce_reversemap.shape[1]:
        #         row = -1
        #     else:
        #         row = self._array_ce_reversemap[c, j]
        row = -1

        if array_av[j]: # and row != -1:

            #     if self.model_quantity_ca_param.shape[0]:
            #         for i in range(self.model_quantity_ca_param.shape[0]):
            #             if row >= 0:
            #                 _temp = self._array_ce[row, self.model_quantity_ca_data[i]]
            #             else:
            #                 _temp = self._array_ca[c, j, self.model_quantity_ca_data[i]]
            #             _temp *= self.model_quantity_ca_param_value[i] * self.model_quantity_ca_param_scale[i]
            #             U[j] += _temp
            #             if not self.model_quantity_ca_param_holdfast[i]:
            #                 dU[j, self.model_quantity_ca_param[i]] += _temp * self.model_quantity_scale_param_value
            #
            #         for i in range(self.model_quantity_ca_param.shape[0]):
            #             if not self.model_quantity_ca_param_holdfast[i]:
            #                 dU[j, self.model_quantity_ca_param[i]] /= U[j]
            #
            #         if Q is not None:
            #             Q[j] = U[j]
            #         IF
            #         DOUBLE_PRECISION:
            #         _temp = log(U[j])
            #     ELSE:
            #     _temp = logf(U[j])
            # U[j] = _temp * self.model_quantity_scale_param_value
            # if (self.model_quantity_scale_param >= 0) and not self.model_quantity_scale_param_holdfast:
            #     dU[j, self.model_quantity_scale_param] += _temp

            for i in range(model_utility_ca_param.shape[0]):
                if row >= 0:
                    _temp = 0.0 #_temp = self._array_ce[row, self.model_utility_ca_data[i]]
                else:
                    _temp = array_ca[j, model_utility_ca_data[i]]
                _temp *= model_utility_ca_param_scale[i]
                utility_elem[j] += _temp * parameter_arr[model_utility_ca_param[i]]
                if not holdfast_arr[model_utility_ca_param[i]]:
                    dutility_elem[j, model_utility_ca_param[i]] += _temp
        else:
            utility_elem[j] = -np.inf


def _type_signature(sig, precision=32):
    result = ()
    for s in sig:
        if s == "f":
            result += (f32[:],) if precision==32 else (f64[:],)
        elif s == "F":
            result += (f32[:,:],) if precision == 32 else (f64[:,:],)
        elif s == "i":
            result += (i32[:],)
        elif s == "b":
            result += (i8[:],)
        elif s == 'B':
            result += (boolean, )
    return result


def _type_signatures(sig):
    return [
        _type_signature(sig, precision=32),
        _type_signature(sig, precision=64),
    ]

_master_shape_signature = (
    '(uca),(uca),(uca), '
    '(uco),(uco),(uco),(uco), '
    '(edges),(edges),(edges),(edges),(nests), '
    '(params),(params), '
    '(nodes),(nodes),(),(vco),(alts,vca), '
    '(),(),()->'
    '(nodes),(nodes),(nodes),(params,params),(params),()'
)

@guvectorize(
    _type_signatures("fii ifii iiiii bf fbffF BBB fffFff"),
    _master_shape_signature,
    nopython=True,
    fastmath=True,
    target='parallel',
    cache=True,
)
def _numba_master(
        model_utility_ca_param_scale,  # [0] float input shape=[n_u_ca_features]
        model_utility_ca_param,        # [1] int input shape=[n_u_ca_features]
        model_utility_ca_data,         # [2] int input shape=[n_u_ca_features]

        model_utility_co_alt,          # [3] int input shape=[n_co_features]
        model_utility_co_param_scale,  # [4] float input shape=[n_co_features]
        model_utility_co_param,        # [5] int input shape=[n_co_features]
        model_utility_co_data,         # [6] int input shape=[n_co_features]

        upslots,       # [7] int input shape=[edges]
        dnslots,       # [8] int input shape=[edges]
        visit1,        # [9] int input shape=[edges]
        allocslot,     # [10] int input shape=[edges]
        mu_slots,      # [11] int input shape=[nests]

        holdfast_arr,  # [12] int8 input shape=[n_params]
        parameter_arr, # [13] float input shape=[n_params]

        array_ch,      # [14] float input shape=[nodes]
        array_av,      # [15] int8 input shape=[nodes]
        array_wt,      # [16] float input shape=[]
        array_co,      # [17] float input shape=[n_co_vars]
        array_ca,      # [18] float input shape=[n_alts, n_ca_vars]

        return_probability,  # [19] bool input
        return_grad,         # [20] bool input
        return_bhhh,         # [21] bool input

        utility,       # [22] float output shape=[nodes]
        logprob,       # [23] float output shape=[nodes]
        probability,   # [24] float output shape=[nodes]
        bhhh,          # [25] float output shape=[n_params, n_params]
        d_loglike,     # [26] float output shape=[n_params]
        loglike,       # [27] float output shape=[]
):
    n_alts = array_ca.shape[0]
    utility[:] = 0.0
    dutility = np.zeros((utility.size, parameter_arr.size), dtype=utility.dtype)

    utility_from_data_ca(
        model_utility_ca_param_scale,  # int input shape=[n_u_ca_features]
        model_utility_ca_param,  # int input shape=[n_u_ca_features]
        model_utility_ca_data,  # int input shape=[n_u_ca_features]
        parameter_arr,  # float input shape=[n_params]
        holdfast_arr,  # float input shape=[n_params]
        array_av,  # int8 input shape=[n_nodes]
        array_ca,  # float input shape=[n_alts, n_ca_vars]
        utility[:n_alts],  # float output shape=[n_alts]
        dutility[:n_alts],
    )

    utility_from_data_co(
        model_utility_co_alt,  # int input shape=[n_co_features]
        model_utility_co_param_scale,  # float input shape=[n_co_features]
        model_utility_co_param,  # int input shape=[n_co_features]
        model_utility_co_data,  # int input shape=[n_co_features]
        parameter_arr,  # float input shape=[n_params]
        holdfast_arr,  # float input shape=[n_params]
        array_av,  # int8 input shape=[n_nodes]
        array_co,  # float input shape=[n_co_vars]
        utility[:n_alts],  # float output shape=[n_alts]
        dutility[:n_alts],
    )

    #util_nx = np.zeros_like(utility)
    #mu_extra = np.zeros_like(util_nx)
    loglike[0] = 0.0

    for s in range(upslots.size):
        dn = dnslots[s]
        up = upslots[s]
        up_nest = up - n_alts
        dn_nest = dn - n_alts
        if mu_slots[up_nest] < 0:
            mu_up = 1.0
        else:
            mu_up = parameter_arr[mu_slots[up_nest]]
        if visit1[s]>0 and dn>=n_alts:
            log_dn = np.log(utility[dn])
            #mu_extra[dn] += log_dn + util_nx[dn]/utility[dn]
            if mu_slots[dn_nest] < 0:
                mu_dn = 1.0
            else:
                mu_dn = parameter_arr[mu_slots[dn_nest]]
            utility[dn] = log_dn * mu_dn
        util_dn = utility[dn]
        exp_util_dn_mu_up = np.exp(util_dn / mu_up)
        utility[up] += exp_util_dn_mu_up
        #util_nx[up] -= util_dn * exp_util_dn_mu_up / mu_up

    #mu_extra[mu_extra.size-1] += np.log(utility[utility.size-1]) + util_nx[-1]/utility[utility.size-1]
    utility[utility.size-1] = np.log(utility[utility.size-1])

    for s in range(upslots.size):
        dn = dnslots[s]
        up = upslots[s]
        if mu_slots[up - n_alts] < 0:
            mu_up = 1.0
        else:
            mu_up = parameter_arr[mu_slots[up - n_alts]]
        logprob[dn] = (utility[dn] - utility[up]) / mu_up
        if array_ch[dn]:
            loglike[0] += logprob[dn] * array_ch[dn] * array_wt[0]

    if return_probability or return_grad or return_bhhh:

        conditional_probability = np.zeros_like(logprob)
        for i in range(logprob.size):
            if array_av[i]:
                conditional_probability[i] = np.exp(logprob[i])

        # probability
        probability[-1] = 1.0
        for s in range(upslots.size-1, -1, -1):
            dn = dnslots[s]
            if array_av[dn]:
                up = upslots[s]
                probability[dn] = probability[up] * conditional_probability[dn]
            else:
                probability[dn] = 0.0

        if return_grad or return_bhhh:

            d_loglike[:] = 0.0

            # d utility
            for s in range(upslots.size):
                dn = dnslots[s]
                up = upslots[s]
                if array_av[dn]:
                    cond_prob = conditional_probability[dn]
                    if dn >= n_alts:
                        dn_mu_slot = mu_slots[dn-n_alts]
                        if dn_mu_slot >= 0:
                            dutility[dn, dn_mu_slot] += utility[dn]
                            dutility[dn, dn_mu_slot] /= parameter_arr[dn_mu_slot]
                    up_mu_slot = mu_slots[up - n_alts]
                    if up_mu_slot >= 0:
                        # FIXME: alpha slots to appear here if cross-nesting is activated
                        dutility[up, up_mu_slot] -= cond_prob * (utility[dn])
                    dutility[up, :] += cond_prob * dutility[dn, :]

            # d probability
            scratch = np.zeros_like(parameter_arr)
            d_probability = np.zeros_like(dutility)
            for s in range(upslots.size-1, -1, -1):
                dn = dnslots[s]
                if array_ch[dn]:
                    up = upslots[s]
                    scratch[:] = dutility[dn] - dutility[up]
                    up_mu_slot = mu_slots[up-n_alts]
                    if up_mu_slot < 0:
                        mu_up = 1.0
                    else:
                        mu_up = parameter_arr[up_mu_slot]
                    if mu_up:
                        if up_mu_slot >= 0:
                            scratch[up_mu_slot] += (utility[up] - utility[dn]) / mu_up
                            # FIXME: alpha slots to appear here if cross-nesting is activated
                        multiplier = probability[up] / mu_up
                    else:
                        multiplier = 0

                    scratch[:] *= multiplier
                    scratch[:] += d_probability[up, :]
                    d_probability[dn, :] += scratch[:] * conditional_probability[dn] # FIXME: for CNL, use edge not dn

            if return_bhhh:
                bhhh[:] = 0.0

            # d loglike
            for a in range(n_alts):
                this_ch = array_ch[a]
                if this_ch == 0:
                    continue
                total_probability_a = probability[a]
                if total_probability_a > 0:
                    ch_over_pr = this_ch / total_probability_a
                    tempvalue = d_probability[a, :] * ch_over_pr
                    dLL_temp = tempvalue / this_ch
                    tempvalue *= array_wt[0]
                    d_loglike += tempvalue
                    if return_bhhh:
                        bhhh += np.outer(dLL_temp,dLL_temp) * this_ch * array_wt[0]


def model_co_slots(dataframes, model, dtype=np.float64):
    len_co = sum(len(_) for _ in model.utility_co.values())
    model_utility_co_alt = np.zeros([len_co], dtype=np.int32)
    model_utility_co_param_scale = np.ones([len_co], dtype=dtype)
    model_utility_co_param = np.zeros([len_co], dtype=np.int32)
    model_utility_co_data = np.zeros([len_co], dtype=np.int32)

    j = 0

    param_loc = {}
    for _n, _pname in enumerate(model._frame.index):
        param_loc[_pname] = _n
    data_loc = {}
    if dataframes.data_co is not None:
        for _n, _dname in enumerate(dataframes.data_co.columns):
            data_loc[_dname] = _n

    alternative_codes = dataframes.alternative_codes()

    for alt, func in model.utility_co.items():
        altindex = alternative_codes.get_loc(alt)
        for i in func:
            model_utility_co_alt[j] = altindex
            model_utility_co_param[j] = param_loc[str(i.param)]  # model._frame.index.get_loc(str(i.param))
            model_utility_co_param_scale[j] = i.scale
            if i.data == '1':
                model_utility_co_data[j] = -1
            else:
                model_utility_co_data[j] = data_loc[str(i.data)]  # self._data_co.columns.get_loc(str(i.data))
            j += 1

    return (
        model_utility_co_alt,
        model_utility_co_param_scale,
        model_utility_co_param,
        model_utility_co_data,
    )


def model_u_ca_slots(dataframes, model, dtype=np.float64):
    len_model_utility_ca = len(model.utility_ca)
    model_utility_ca_param_scale = np.ones([len_model_utility_ca], dtype=dtype)
    model_utility_ca_param = np.zeros([len_model_utility_ca], dtype=np.int32)
    model_utility_ca_data = np.zeros([len_model_utility_ca], dtype=np.int32)
    for n, i in enumerate(model.utility_ca):
        model_utility_ca_param[n] = model._frame.index.get_loc(str(i.param))
        model_utility_ca_data[n] = dataframes.data_ca_or_ce.columns.get_loc(str(i.data))
        model_utility_ca_param_scale[n] = i.scale
    return (
        model_utility_ca_param_scale,
        model_utility_ca_param,
        model_utility_ca_data,
    )

from collections import namedtuple
WorkArrays = namedtuple(
    'WorkArrays',
    ['utility', 'logprob', 'probability', 'bhhh', 'd_loglike', 'loglike'],
)



class NumbaModel(_BaseModel):

    def __init__(self, *args, float_dtype = np.float64, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_arrays = None
        self.work_arrays = None
        self.float_dtype = float_dtype

    def mangle(self, *args, **kwargs):
        super().mangle(*args, **kwargs)
        self._fixed_arrays = None
        self.work_arrays = None
        self._array_ch_cascade = None
        self._array_av_cascade = None

    def unmangle(self, force=False):
        super().unmangle(force=force)
        if self._fixed_arrays is None:
            n_nodes = len(self.graph)
            n_alts = self.graph.n_elementals()
            n_nests = n_nodes - n_alts
            n_params = len(self._frame)
            if self.dataframes is not None:
                (
                    model_utility_ca_param_scale,
                    model_utility_ca_param,
                    model_utility_ca_data,
                ) = model_u_ca_slots(self.dataframes, self, dtype=self.float_dtype)
                (
                    model_utility_co_alt,
                    model_utility_co_param_scale,
                    model_utility_co_param,
                    model_utility_co_data,
                ) = model_co_slots(self.dataframes, self, dtype=self.float_dtype)

                node_slot_arrays = self.graph.node_slot_arrays(self)

                self._fixed_arrays = (
                    model_utility_ca_param_scale,
                    model_utility_ca_param,
                    model_utility_ca_data,

                    model_utility_co_alt,
                    model_utility_co_param_scale,
                    model_utility_co_param,
                    model_utility_co_data,

                    *self.graph.edge_slot_arrays(),
                    node_slot_arrays[0][n_alts:],
                )

                if self.dataframes.data_ch is not None:
                    _array_ch_cascade = self.dataframes.data_ch_cascade(self.graph).to_numpy()
                else:
                    _array_ch_cascade = np.empty([self.n_cases, 0], dtype=self.float_dtype)
                if self.dataframes.data_av is not None:
                    _array_av_cascade = self.dataframes.data_av_cascade(self.graph).to_numpy()
                else:
                    _array_av_cascade = np.ones([self.n_cases, n_nodes], dtype=np.int8)
                if self.dataframes.data_wt is not None:
                    _array_wt = self.dataframes.array_wt().astype(self.float_dtype)
                else:
                    _array_wt = self.float_dtype(1.0)

                self._data_arrays = (
                    (_array_ch_cascade.astype(self.float_dtype)),
                    (_array_av_cascade),
                    (_array_wt.reshape(-1)),
                    (self.dataframes.array_co(force=True).astype(self.float_dtype)),
                    (self.dataframes.array_ca(force=True).astype(self.float_dtype)),
                )
            else:
                self._fixed_arrays = None
                self._data_arrays = None

            try:
                n_cases = self.n_cases
            except MissingDataError:
                self.work_arrays = None
            else:
                self.work_arrays = WorkArrays(
                    utility=np.zeros([self.n_cases, n_nodes], dtype=self.float_dtype),
                    logprob=np.zeros([self.n_cases, n_nodes], dtype=self.float_dtype),
                    probability=np.zeros([self.n_cases, n_nodes], dtype=self.float_dtype),
                    bhhh=np.zeros([self.n_cases, n_params, n_params], dtype=self.float_dtype),
                    d_loglike=np.zeros([self.n_cases, n_params], dtype=self.float_dtype),
                    loglike=np.zeros([self.n_cases], dtype=self.float_dtype),
                )

    def __prepare_for_compute(
            self,
            x=None,
            allow_missing_ch=False,
            allow_missing_av=False,
    ):
        missing_ch, missing_av = False, False
        if self.dataframes is None:
            raise MissingDataError('dataframes is not set, maybe you need to call `load_data` first?')
        if not self.dataframes.is_computational_ready(activate=True):
            raise ValueError('DataFrames is not computational-ready')
        if x is not None:
            self.set_values(x)
        self.unmangle()
        if self.dataframes.data_ch is None:
            if allow_missing_ch:
                missing_ch = True
            else:
                raise MissingDataError('model.dataframes does not define data_ch')
        if self.dataframes.data_av is None:
            if allow_missing_av:
                missing_av = True
            else:
                raise MissingDataError('model.dataframes does not define data_av')
        return (
            *self._fixed_arrays,
            self._frame.holdfast.to_numpy(),
            self.pvals.astype(self.float_dtype), # float input shape=[n_params]
            *self._data_arrays,
        )

    def _loglike_runner(
            self,
            x=None,
            return_gradient=False,
            return_probability=False,
            return_bhhh=False,
    ):
        args = self.__prepare_for_compute(x)
        try:
            result_arrays = WorkArrays(*_numba_master(
                *args,
                return_probability,
                return_gradient,
                return_bhhh,
                out=tuple(self.work_arrays),
            ))
        except:
            shp = lambda y: getattr(y, 'shape', 'scalar')
            dtp = lambda y: getattr(y, 'dtype', 'untyped')

            in_sig, out_sig = _master_shape_signature.split("->")
            print("# Input Arrays")
            for n, (a, s) in enumerate(zip(args, in_sig.split("(")[1:])):
                s = s.rstrip(" ),")
                print(f" [{n:2}] {s.strip():9}: {dtp(a)}{shp(a)}")
            print("# Output Arrays")
            for n, (a, s) in enumerate(zip(self.work_arrays, out_sig.split("(")[1:])):
                s = s.rstrip(" ),")
                print(f" [{n:2}] {s.strip():9}: {dtp(a)}{shp(a)}")
            raise
        return result_arrays

    def loglike(self, x=None, *args, **kwargs):
        result_arrays = self._loglike_runner(x)
        return result_arrays.loglike.sum() * self.dataframes.weight_normalization

    def d_loglike(self, x=None, *args, return_series=False, **kwargs):
        result_arrays = self._loglike_runner(x, return_gradient=True)
        result = result_arrays.d_loglike.sum(0) * self.dataframes.weight_normalization
        if return_series:
            result = pd.Series(result, index=self._frame.index)
        return result

    def loglike_casewise(self, x=None, *args, **kwargs):
        result_arrays = self._loglike_runner(x)
        return result_arrays.loglike * self.dataframes.weight_normalization

    def d_loglike_casewise(self, x=None, *args, **kwargs):
        result_arrays = self._loglike_runner(x, return_gradient=True)
        return result_arrays.d_loglike * self.dataframes.weight_normalization

    def bhhh(self, x=None, *args, return_dataframe=False, **kwargs):
        result_arrays = self._loglike_runner(x, return_bhhh=True)
        result = result_arrays.bhhh.sum(0) * self.dataframes.weight_normalization
        if return_dataframe:
            result = pd.DataFrame(
                result, columns=self._frame.index, index=self._frame.index
            )
        return result

    def probability(self, x=None, *args, **kwargs):
        result_arrays = self._loglike_runner(x, return_probability=True)
        return result_arrays.probability

    def loglike2(
            self,
            x=None,
            *,
            start_case=0,
            stop_case=-1,
            step_case=1,
            persist=0,
            leave_out=-1,
            keep_only=-1,
            subsample=-1,
            return_series=True,
            probability_only=False,
    ):
        result_arrays = self._loglike_runner(x, return_gradient=True)
        result = dictx(
            ll=result_arrays.loglike.sum() * self.dataframes.weight_normalization,
            dll=result_arrays.d_loglike.sum(0) * self.dataframes.weight_normalization,
        )
        if start_case == 0 and stop_case == -1 and step_case == 1:
            self._check_if_best(result.ll)
        if return_series:
            result['dll'] = pd.Series(result['dll'], index=self._frame.index, )
        return result

    def loglike2_bhhh(
            self,
            x=None,
            *,
            return_series=False,
            start_case=0, stop_case=-1, step_case=1,
            persist=0,
            leave_out=-1, keep_only=-1, subsample=-1,
    ):
        result_arrays = self._loglike_runner(x, return_gradient=True, return_bhhh=True)
        result = dictx(
            ll=result_arrays.loglike.sum() * self.dataframes.weight_normalization,
            dll=result_arrays.d_loglike.sum(0) * self.dataframes.weight_normalization,
            bhhh=result_arrays.bhhh.sum(0) * self.dataframes.weight_normalization,
        )
        from ..model.persist_flags import PERSIST_LOGLIKE_CASEWISE, PERSIST_D_LOGLIKE_CASEWISE
        if persist & PERSIST_LOGLIKE_CASEWISE:
            result['ll_casewise'] = result_arrays.loglike * self.dataframes.weight_normalization
        if persist & PERSIST_D_LOGLIKE_CASEWISE:
            result['dll_casewise'] = result_arrays.d_loglike * self.dataframes.weight_normalization
        if start_case == 0 and stop_case == -1 and step_case == 1:
            self._check_if_best(result.ll)
        if return_series:
            result['dll'] = pd.Series(result['dll'], index=self._frame.index, )
            result['bhhh'] = pd.DataFrame(
                result['bhhh'], index=self._frame.index, columns=self._frame.index
            )
        return result
