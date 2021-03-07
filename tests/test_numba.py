from larch.examples import MTC, SWISSMETRO
from pytest import approx, raises, warns, fixture
import numpy as np
import pandas as pd
from larch.roles import P, X, PX
from larch.model.persist_flags import PERSIST_UTILITY
from larch.numba.model import NumbaModel


@fixture
def mtc():
    d = MTC()
    df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', )
    df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
    df_av = d.dataframe_idca('_avail_', dtype=bool)
    df_ch = d.dataframe_idca('_choice_')
    from larch.dataframes import DataFrames
    return DataFrames(
        co=df_co,
        ca=df_ca,
        av=df_av,
        ch=df_ch,
    )


def test_dataframes_mnl5(mtc):
    m5 = NumbaModel()

    from larch.roles import P, X, PX

    m5.utility_co[2] = P("ASC_SR2") * X("1") + P("hhinc#2") * X("hhinc")
    m5.utility_co[3] = P("ASC_SR3P") * X("1") + P("hhinc#3") * X("hhinc")
    m5.utility_co[4] = P("ASC_TRAN") * X("1") + P("hhinc#4") * X("hhinc")
    m5.utility_co[5] = P("ASC_BIKE") * X("1") + P("hhinc#5") * X("hhinc")
    m5.utility_co[6] = P("ASC_WALK") * X("1") + P("hhinc#6") * X("hhinc")
    m5.utility_ca = PX("tottime") + PX("totcost")

    m5.dataframes = mtc

    beta_in1 = {
        'ASC_BIKE': -0.8523646111088327,
        'ASC_SR2': -0.5233769323949348,
        'ASC_SR3P': -2.3202089848081027,
        'ASC_TRAN': -0.05615933557609158,
        'ASC_WALK': 0.050082767550586924,
        'hhinc#2': -0.001040241396513087,
        'hhinc#3': 0.0031822969445656542,
        'hhinc#4': -0.0017162484345735326,
        'hhinc#5': -0.004071521055900851,
        'hhinc#6': -0.0021316332241034445,
        'totcost': -0.001336661560553717,
        'tottime': -0.01862990704919887,
    }

    ll2 = m5.loglike2(beta_in1, return_series=True)

    q1_dll = {
        'ASC_BIKE': -139.43832,
        'ASC_SR2': -788.00574,
        'ASC_SR3P': -126.84879,
        'ASC_TRAN': -357.75186,
        'ASC_WALK': -116.137886,
        'hhinc#2': -46416.28,
        'hhinc#3': -8353.63,
        'hhinc#4': -21409.012,
        'hhinc#5': -8299.654,
        'hhinc#6': -7395.375,
        'totcost': 39520.043,
        'tottime': -26556.303,
    }

    assert -4930.3212890625 == approx(ll2.ll)

    for k in q1_dll:
        assert q1_dll[k] == approx(dict(ll2.dll)[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict(ll2.dll)[k]}"

    # Test calculate_parameter_covariance doesn't choke if all holdfasts are on:
    m5.lock_values(*beta_in1.keys())
    m5.calculate_parameter_covariance()

    assert np.all(m5.pf['std_err'] == 0)
    assert np.all(m5.pf['robust_std_err'] == 0)


def test_dataframes_mnl5_ca(mtc):
    m5 = NumbaModel()

    from larch.roles import P, X, PX
    m5.utility_ca = PX("tottime") + PX("totcost")

    m5.dataframes = mtc

    beta_in1 = {
        'totcost': -0.001336661560553717,
        'tottime': -0.01862990704919887,
    }

    ll2 = m5.loglike2(beta_in1, return_series=True)

    q1_dll = {
        'totcost': 173090.625000,
        'tottime': -24771.804688,
    }

    assert -6904.966796875 == approx(ll2.ll, rel=1e-5)

    for k in q1_dll:
        assert q1_dll[k] == approx(dict(ll2.dll)[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict(ll2.dll)[k]}"


def test_dataframes_mnl5_co(mtc):
    m5 = NumbaModel()

    from larch.roles import P, X, PX
    m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
    m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
    m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
    m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
    m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
    m5.dataframes = mtc

    beta_in1 = {
        'ASC_BIKE': -0.8523646111088327,
        'ASC_SR2': -0.5233769323949348,
        'ASC_SR3P': -2.3202089848081027,
        'ASC_TRAN': -0.05615933557609158,
        'ASC_WALK': 0.050082767550586924,
        'hhinc#2': -0.001040241396513087,
        'hhinc#3': 0.0031822969445656542,
        'hhinc#4': -0.0017162484345735326,
        'hhinc#5': -0.004071521055900851,
        'hhinc#6': -0.0021316332241034445,
    }

    ll2 = m5.loglike2(beta_in1, return_series=True)

    q1_dll = {
        'ASC_BIKE': -139.2947,
        'ASC_SR2': -598.531,
        'ASC_SR3P': -77.68647,
        'ASC_TRAN': -715.4206,
        'ASC_WALK': -235.8408,
        'hhinc#2': -35611.855,
        'hhinc#3': -5276.0254,
        'hhinc#4': -42263.88,
        'hhinc#5': -8355.174,
        'hhinc#6': -13866.567,
    }

    assert -5594.70654296875 == approx(ll2.ll, rel=1e-5)

    for k in q1_dll:
        assert q1_dll[k] == approx(dict(ll2.dll)[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict(ll2.dll)[k]}"


def test_dataframes_nl5(mtc):
    m5 = NumbaModel()

    m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
    m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
    m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
    m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
    m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
    m5.utility_ca = PX("tottime") + PX("totcost")

    m5.dataframes = mtc

    m5.graph.add_node(9, children=(5, 6), parameter='MU_NonMotorized')

    beta_in1 = {
        'ASC_BIKE': -0.8523646111088327,
        'ASC_SR2': -0.5233769323949348,
        'ASC_SR3P': -2.3202089848081027,
        'ASC_TRAN': -0.05615933557609158,
        'ASC_WALK': 0.050082767550586924,
        'hhinc#2': -0.001040241396513087,
        'hhinc#3': 0.0031822969445656542,
        'hhinc#4': -0.0017162484345735326,
        'hhinc#5': -0.004071521055900851,
        'hhinc#6': -0.0021316332241034445,
        'totcost': -0.001336661560553717,
        'tottime': -0.01862990704919887,
    }

    ll2 = m5.loglike2(beta_in1, return_series=True)

    q1_dll = {
        'ASC_BIKE': -139.43832,
        'ASC_SR2': -788.00574,
        'ASC_SR3P': -126.84879,
        'ASC_TRAN': -357.75186,
        'ASC_WALK': -116.137886,
        'hhinc#2': -46416.28,
        'hhinc#3': -8353.63,
        'hhinc#4': -21409.012,
        'hhinc#5': -8299.654,
        'hhinc#6': -7395.375,
        'totcost': 39520.043,
        'tottime': -26556.303,
    }

    assert approx(ll2.ll) == -4930.3212890625
    dict_ll2_dll = dict(ll2.dll)

    for k in q1_dll:
        assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict_ll2_dll[k]}"

    beta_in1 = {
        'ASC_BIKE': -0.8523646111088327,
        'ASC_SR2': -0.5233769323949348,
        'ASC_SR3P': -2.3202089848081027,
        'ASC_TRAN': -0.05615933557609158,
        'ASC_WALK': 0.050082767550586924,
        'hhinc#2': -0.001040241396513087,
        'hhinc#3': 0.0031822969445656542,
        'hhinc#4': -0.0017162484345735326,
        'hhinc#5': -0.004071521055900851,
        'hhinc#6': -0.0021316332241034445,
        'totcost': -0.001336661560553717,
        'tottime': -0.01862990704919887,
        'MU_NonMotorized': 0.5,
    }

    ll2b = m5.loglike2(beta_in1, return_series=True)

    q1_dllb = {
        'ASC_BIKE': -94.071343,
        'ASC_SR2': -800.092341,
        'ASC_SR3P': -129.354567,
        'ASC_TRAN': -369.808551,
        'ASC_WALK': -114.786728,
        'MU_NonMotorized': -34.816070,
        'hhinc#2': -47089.611079,
        'hhinc#3': -8505.116916,
        'hhinc#4': -22071.859018,
        'hhinc#5': -5844.969336,
        'hhinc#6': -7168.859044,
        'totcost': 37322.528282,
        'tottime': -26479.290942,
    }

    assert -4897.764630665653 == approx(ll2b.ll)

    dict_ll2b_dll = dict(ll2b.dll)

    for k in q1_dllb:
        assert q1_dllb[k] == approx(dict_ll2b_dll[k], rel=1e-5), f"{k} {q1_dllb[k]} != {dict_ll2b_dll[k]}"

    print(m5.check_d_loglike().data['similarity'].min())
    chk = m5.check_d_loglike()

    assert chk.data['similarity'].min() > 4


def test_weighted_bhhh():
    d = MTC()

    df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', )
    df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
    df_av = d.dataframe_idca('_avail_', dtype=bool)
    df_ch = d.dataframe_idca('_choice_')

    df_co2 = pd.concat([df_co, df_co]).reset_index(drop=True)
    df_ca2 = pd.concat([df_ca.unstack(), df_ca.unstack()]).reset_index(drop=True).stack()
    df_av2 = pd.concat([df_av.unstack(), df_av.unstack()]).reset_index(drop=True).stack()

    df_chX = pd.DataFrame(
        np.zeros_like(df_ch.values),
        index=df_ch.index,
        columns=df_ch.columns,
    )
    df_chX = df_chX.unstack()
    df_chX.iloc[:, 1] = 2.0
    df_chX = df_chX.stack()

    df_ch2 = pd.concat([df_ch.unstack(), df_chX.unstack()]).reset_index(drop=True).stack()

    from larch import DataFrames, Model

    j1 = DataFrames(
        co=df_co,
        ca=df_ca,
        av=df_av,
        ch=df_chX + df_ch,
    )

    j2 = DataFrames(
        co=df_co2,
        ca=df_ca2,
        av=df_av2,
        ch=df_ch2,
    )

    j1.autoscale_weights()
    j2.autoscale_weights()

    m5 = NumbaModel()

    from larch.roles import P, X, PX
    m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
    m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
    m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
    m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
    m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
    m5.utility_ca = PX("tottime") + PX("totcost")

    beta_in1 = {
        'ASC_BIKE': -0.8523646111088327,
        'ASC_SR2': -0.5233769323949348,
        'ASC_SR3P': -2.3202089848081027,
        'ASC_TRAN': -0.05615933557609158,
        'ASC_WALK': 0.050082767550586924,
        'hhinc#2': -0.001040241396513087,
        'hhinc#3': 0.0031822969445656542,
        'hhinc#4': -0.0017162484345735326,
        'hhinc#5': -0.004071521055900851,
        'hhinc#6': -0.0021316332241034445,
        'totcost': -0.001336661560553717,
        'tottime': -0.01862990704919887,
    }


    m5.dataframes = j1
    m5.pf_sort()
    ll1 = m5.loglike(beta_in1)
    dll1 = m5.d_loglike(beta_in1, return_series=True)
    bhhh1 = m5.bhhh(beta_in1, return_dataframe=True)

    m5.dataframes = j2
    m5.pf_sort()
    ll2 = m5.loglike(beta_in1)
    dll2 = m5.d_loglike(beta_in1, return_series=True)
    bhhh2 = m5.bhhh(beta_in1, return_dataframe=True)

    q1_dll = {
        'ASC_BIKE': -518.3145850719714,
        'ASC_SR2': 6659.9870966633935,
        'ASC_SR3P': -702.5461471592637,
        'ASC_TRAN': -2069.2556854096474,
        'ASC_WALK': -680.4136747673049,
        'hhinc#2': 390300.04704708763,
        'hhinc#3': -44451.89987844542,
        'hhinc#4': -117769.88300441334,
        'hhinc#5': -29774.93396444093,
        'hhinc#6': -36754.12651709895,
        'totcost': -280658.27799924824,
        'tottime': -66172.15328009706,
    }

    assert (j1.weight_normalization, j2.weight_normalization) == (3.0, 1.5)

    assert (ll1, ll2) == approx((-18829.858031378415, -18829.858031378433))
    dict_ll1_dll = dict(dll1)
    dict_ll2_dll = dict(dll2)

    for k in q1_dll:
        assert q1_dll[k] == approx(dict_ll1_dll[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict_ll1_dll[k]}"
        assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict_ll2_dll[k]}"

    bhhh_correct = {
        ('ASC_BIKE', 'ASC_BIKE'): 102.45820678523617,
        ('ASC_BIKE', 'ASC_SR2'): -285.118579955482,
        ('ASC_BIKE', 'ASC_SR3P'): 19.760852749005203,
        ('ASC_BIKE', 'ASC_TRAN'): 76.57398723818505,
        ('ASC_BIKE', 'ASC_WALK'): 32.34845311016433,
        ('ASC_BIKE', 'hhinc#2'): -16063.872652571783,
        ('ASC_BIKE', 'hhinc#3'): 1231.2989480753972,
        ('ASC_BIKE', 'hhinc#4'): 4436.148875125188,
        ('ASC_BIKE', 'hhinc#5'): 5322.601183863066,
        ('ASC_BIKE', 'hhinc#6'): 1870.2419763938155,
        ('ASC_BIKE', 'totcost'): 651.7010091466211,
        ('ASC_BIKE', 'tottime'): 3410.0934222585456,
        ('ASC_SR2', 'ASC_BIKE'): -285.118579955482,
        ('ASC_SR2', 'ASC_SR2'): 6156.860848428149,
        ('ASC_SR2', 'ASC_SR3P'): -412.00155512537174,
        ('ASC_SR2', 'ASC_TRAN'): -1312.7312255613908,
        ('ASC_SR2', 'ASC_WALK'): -461.36148965146094,
        ('ASC_SR2', 'hhinc#2'): 360960.5064929568,
        ('ASC_SR2', 'hhinc#3'): -25836.407171358598,
        ('ASC_SR2', 'hhinc#4'): -73800.95798232155,
        ('ASC_SR2', 'hhinc#5'): -16063.872652571783,
        ('ASC_SR2', 'hhinc#6'): -23821.20824758219,
        ('ASC_SR2', 'totcost'): -255314.72401330443,
        ('ASC_SR2', 'tottime'): -26043.72175056138,
        ('ASC_SR3P', 'ASC_BIKE'): 19.760852749005203,
        ('ASC_SR3P', 'ASC_SR2'): -412.00155512537174,
        ('ASC_SR3P', 'ASC_SR3P'): 194.01812799791645,
        ('ASC_SR3P', 'ASC_TRAN'): 75.72048947864153,
        ('ASC_SR3P', 'ASC_WALK'): 21.268891825845834,
        ('ASC_SR3P', 'hhinc#2'): -25836.407171358598,
        ('ASC_SR3P', 'hhinc#3'): 11951.689201021733,
        ('ASC_SR3P', 'hhinc#4'): 4654.093182472213,
        ('ASC_SR3P', 'hhinc#5'): 1231.2989480753972,
        ('ASC_SR3P', 'hhinc#6'): 1291.5421535124271,
        ('ASC_SR3P', 'totcost'): -873.8405986324893,
        ('ASC_SR3P', 'tottime'): 2847.414853230761,
        ('ASC_TRAN', 'ASC_BIKE'): 76.57398723818505,
        ('ASC_TRAN', 'ASC_SR2'): -1312.7312255613908,
        ('ASC_TRAN', 'ASC_SR3P'): 75.72048947864153,
        ('ASC_TRAN', 'ASC_TRAN'): 795.8802149505715,
        ('ASC_TRAN', 'ASC_WALK'): 106.98212315599287,
        ('ASC_TRAN', 'hhinc#2'): -73800.95798232155,
        ('ASC_TRAN', 'hhinc#3'): 4654.093182472214,
        ('ASC_TRAN', 'hhinc#4'): 43744.97724907039,
        ('ASC_TRAN', 'hhinc#5'): 4436.148875125188,
        ('ASC_TRAN', 'hhinc#6'): 5766.003794801968,
        ('ASC_TRAN', 'totcost'): -289.2663620264443,
        ('ASC_TRAN', 'tottime'): 17843.995404571946,
        ('ASC_WALK', 'ASC_BIKE'): 32.34845311016433,
        ('ASC_WALK', 'ASC_SR2'): -461.36148965146094,
        ('ASC_WALK', 'ASC_SR3P'): 21.268891825845834,
        ('ASC_WALK', 'ASC_TRAN'): 106.98212315599287,
        ('ASC_WALK', 'ASC_WALK'): 257.42682167907014,
        ('ASC_WALK', 'hhinc#2'): -23821.20824758219,
        ('ASC_WALK', 'hhinc#3'): 1291.5421535124271,
        ('ASC_WALK', 'hhinc#4'): 5766.003794801967,
        ('ASC_WALK', 'hhinc#5'): 1870.2419763938155,
        ('ASC_WALK', 'hhinc#6'): 12476.241347103956,
        ('ASC_WALK', 'totcost'): -4472.049603002317,
        ('ASC_WALK', 'tottime'): 7147.124392574634,
        ('hhinc#2', 'ASC_BIKE'): -16063.872652571783,
        ('hhinc#2', 'ASC_SR2'): 360960.5064929568,
        ('hhinc#2', 'ASC_SR3P'): -25836.407171358598,
        ('hhinc#2', 'ASC_TRAN'): -73800.95798232155,
        ('hhinc#2', 'ASC_WALK'): -23821.20824758219,
        ('hhinc#2', 'hhinc#2'): 27739015.863625936,
        ('hhinc#2', 'hhinc#3'): -2107887.1245679897,
        ('hhinc#2', 'hhinc#4'): -5551797.986970257,
        ('hhinc#2', 'hhinc#5'): -1180261.954185782,
        ('hhinc#2', 'hhinc#6'): -1731206.5786703676,
        ('hhinc#2', 'totcost'): -15915701.570008647,
        ('hhinc#2', 'tottime'): -1404099.9397647786,
        ('hhinc#3', 'ASC_BIKE'): 1231.2989480753972,
        ('hhinc#3', 'ASC_SR2'): -25836.407171358598,
        ('hhinc#3', 'ASC_SR3P'): 11951.689201021733,
        ('hhinc#3', 'ASC_TRAN'): 4654.093182472214,
        ('hhinc#3', 'ASC_WALK'): 1291.5421535124271,
        ('hhinc#3', 'hhinc#2'): -2107887.1245679897,
        ('hhinc#3', 'hhinc#3'): 985365.3310746389,
        ('hhinc#3', 'hhinc#4'): 365082.18492341647,
        ('hhinc#3', 'hhinc#5'): 96863.5554530797,
        ('hhinc#3', 'hhinc#6'): 105646.64587551638,
        ('hhinc#3', 'totcost'): -18197.967760858926,
        ('hhinc#3', 'tottime'): 173393.88742844874,
        ('hhinc#4', 'ASC_BIKE'): 4436.148875125188,
        ('hhinc#4', 'ASC_SR2'): -73800.95798232155,
        ('hhinc#4', 'ASC_SR3P'): 4654.093182472213,
        ('hhinc#4', 'ASC_TRAN'): 43744.97724907039,
        ('hhinc#4', 'ASC_WALK'): 5766.003794801967,
        ('hhinc#4', 'hhinc#2'): -5551797.986970257,
        ('hhinc#4', 'hhinc#3'): 365082.18492341647,
        ('hhinc#4', 'hhinc#4'): 3238425.3752979557,
        ('hhinc#4', 'hhinc#5'): 328415.2849030458,
        ('hhinc#4', 'hhinc#6'): 431510.210405761,
        ('hhinc#4', 'totcost'): -203942.90089356547,
        ('hhinc#4', 'tottime'): 996131.6491729214,
        ('hhinc#5', 'ASC_BIKE'): 5322.601183863066,
        ('hhinc#5', 'ASC_SR2'): -16063.872652571783,
        ('hhinc#5', 'ASC_SR3P'): 1231.2989480753972,
        ('hhinc#5', 'ASC_TRAN'): 4436.148875125188,
        ('hhinc#5', 'ASC_WALK'): 1870.2419763938155,
        ('hhinc#5', 'hhinc#2'): -1180261.954185782,
        ('hhinc#5', 'hhinc#3'): 96863.5554530797,
        ('hhinc#5', 'hhinc#4'): 328415.2849030458,
        ('hhinc#5', 'hhinc#5'): 376816.6222240454,
        ('hhinc#5', 'hhinc#6'): 139543.81527069444,
        ('hhinc#5', 'totcost'): 64871.45953365492,
        ('hhinc#5', 'tottime'): 191184.20323081187,
        ('hhinc#6', 'ASC_BIKE'): 1870.2419763938155,
        ('hhinc#6', 'ASC_SR2'): -23821.20824758219,
        ('hhinc#6', 'ASC_SR3P'): 1291.5421535124271,
        ('hhinc#6', 'ASC_TRAN'): 5766.003794801968,
        ('hhinc#6', 'ASC_WALK'): 12476.241347103956,
        ('hhinc#6', 'hhinc#2'): -1731206.5786703676,
        ('hhinc#6', 'hhinc#3'): 105646.64587551638,
        ('hhinc#6', 'hhinc#4'): 431510.210405761,
        ('hhinc#6', 'hhinc#5'): 139543.81527069444,
        ('hhinc#6', 'hhinc#6'): 872604.6192807176,
        ('hhinc#6', 'totcost'): -95081.80795513398,
        ('hhinc#6', 'tottime'): 364196.75026433205,
        ('totcost', 'ASC_BIKE'): 651.7010091466211,
        ('totcost', 'ASC_SR2'): -255314.72401330443,
        ('totcost', 'ASC_SR3P'): -873.8405986324893,
        ('totcost', 'ASC_TRAN'): -289.2663620264443,
        ('totcost', 'ASC_WALK'): -4472.049603002317,
        ('totcost', 'hhinc#2'): -15915701.570008647,
        ('totcost', 'hhinc#3'): -18197.967760858926,
        ('totcost', 'hhinc#4'): -203942.90089356547,
        ('totcost', 'hhinc#5'): 64871.45953365492,
        ('totcost', 'hhinc#6'): -95081.80795513398,
        ('totcost', 'totcost'): 67516567.00440338,
        ('totcost', 'tottime'): -775245.3645765022,
        ('tottime', 'ASC_BIKE'): 3410.0934222585456,
        ('tottime', 'ASC_SR2'): -26043.72175056138,
        ('tottime', 'ASC_SR3P'): 2847.414853230761,
        ('tottime', 'ASC_TRAN'): 17843.995404571946,
        ('tottime', 'ASC_WALK'): 7147.124392574634,
        ('tottime', 'hhinc#2'): -1404099.9397647786,
        ('tottime', 'hhinc#3'): 173393.88742844874,
        ('tottime', 'hhinc#4'): 996131.6491729214,
        ('tottime', 'hhinc#5'): 191184.20323081187,
        ('tottime', 'hhinc#6'): 364196.75026433205,
        ('tottime', 'totcost'): -775245.3645765022,
        ('tottime', 'tottime'): 910724.9012464394,
    }

    assert dict(bhhh1.unstack()) == approx(bhhh_correct)
    assert dict(bhhh2.unstack()) == approx(bhhh_correct)

    assert m5.check_d_loglike().data.similarity.min() > 4


def test_weighted_nl_bhhh():
    d = MTC()

    df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', )
    df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
    df_av = d.dataframe_idca('_avail_', dtype=bool)
    df_ch = d.dataframe_idca('_choice_')

    df_co2 = pd.concat([df_co, df_co]).reset_index(drop=True)
    df_ca2 = pd.concat([df_ca.unstack(), df_ca.unstack()]).reset_index(drop=True).stack()
    df_av2 = pd.concat([df_av.unstack(), df_av.unstack()]).reset_index(drop=True).stack()

    df_chX = pd.DataFrame(
        np.zeros_like(df_ch.values),
        index=df_ch.index,
        columns=df_ch.columns,
    )
    df_chX = df_chX.unstack()
    df_chX.iloc[:, 1] = 2.0
    df_chX = df_chX.stack()

    df_ch2 = pd.concat([df_ch.unstack(), df_chX.unstack()]).reset_index(drop=True).stack()

    from larch import DataFrames, Model

    j1 = DataFrames(
        co=df_co,
        ca=df_ca,
        av=df_av,
        ch=df_chX + df_ch,
    )

    j2 = DataFrames(
        co=df_co2,
        ca=df_ca2,
        av=df_av2,
        ch=df_ch2,
    )

    j1.autoscale_weights()
    j2.autoscale_weights()

    m5 = NumbaModel()

    from larch.roles import P, X, PX
    m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
    m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
    m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
    m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
    m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
    m5.utility_ca = PX("tottime") + PX("totcost")

    m5.initialize_graph(alternative_codes=[1, 2, 3, 4, 5, 6])
    m5.graph.add_node(10, children=(5, 6), parameter='MU_nonmotor')
    m5.graph.add_node(11, children=(1, 2, 3), parameter='MU_car')

    beta_in1 = {
        'ASC_BIKE': -0.8523646111088327,
        'ASC_SR2': -0.5233769323949348,
        'ASC_SR3P': -2.3202089848081027,
        'ASC_TRAN': -0.05615933557609158,
        'ASC_WALK': 0.050082767550586924,
        'hhinc#2': -0.001040241396513087,
        'hhinc#3': 0.0031822969445656542,
        'hhinc#4': -0.0017162484345735326,
        'hhinc#5': -0.004071521055900851,
        'hhinc#6': -0.0021316332241034445,
        'totcost': -0.001336661560553717,
        'tottime': -0.01862990704919887,
    }

    m5.pf_sort()

    m5.dataframes = j1
    ll1 = m5.loglike2(beta_in1)
    dll1 = m5.d_loglike(beta_in1, return_series=True)
    bhhh1 = m5.bhhh(beta_in1, return_dataframe=True)

    from larch.model import PERSIST_ALL
    ll1 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=PERSIST_ALL)

    m5.dataframes = j2
    m5.mangle()
    ll2 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=PERSIST_ALL)

    q1_dll = {
        'ASC_BIKE': -518.3145850719714,
        'ASC_SR2': 6659.9870966633935,
        'ASC_SR3P': -702.5461471592637,
        'ASC_TRAN': -2069.2556854096474,
        'ASC_WALK': -680.4136747673049,
        'hhinc#2': 390300.04704708763,
        'hhinc#3': -44451.89987844542,
        'hhinc#4': -117769.88300441334,
        'hhinc#5': -29774.93396444093,
        'hhinc#6': -36754.12651709895,
        'totcost': -280658.27799924824,
        'tottime': -66172.15328009706,
    }

    assert (j1.weight_normalization, j2.weight_normalization) == (3.0, 1.5)
    assert (ll1.ll, ll2.ll) == approx((-18829.858031378415, -18829.858031378433))
    dict_ll1_dll = dict(ll1.dll)
    dict_ll2_dll = dict(ll2.dll)
    for k in q1_dll:
        assert q1_dll[k] == approx(dict_ll1_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll1_dll[k]}"
        assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll2_dll[k]}"

    bhhh_correct = {
        ('ASC_BIKE', 'ASC_BIKE'): 102.45820678523614,
        ('ASC_BIKE', 'ASC_SR2'): -285.118579955482,
        ('ASC_BIKE', 'ASC_SR3P'): 19.760852749005203,
        ('ASC_BIKE', 'ASC_TRAN'): 76.573987238185,
        ('ASC_BIKE', 'ASC_WALK'): 32.348453110164314,
        ('ASC_BIKE', 'MU_car'): -219.27982884201612,
        ('ASC_BIKE', 'MU_nonmotor'): 69.38016083689357,
        ('ASC_BIKE', 'hhinc#2'): -16063.872652571805,
        ('ASC_BIKE', 'hhinc#3'): 1231.2989480753972,
        ('ASC_BIKE', 'hhinc#4'): 4436.148875125189,
        ('ASC_BIKE', 'hhinc#5'): 5322.601183863066,
        ('ASC_BIKE', 'hhinc#6'): 1870.2419763938155,
        ('ASC_BIKE', 'totcost'): 651.7010091466225,
        ('ASC_BIKE', 'tottime'): 3410.093422258546,
        ('ASC_SR2', 'ASC_BIKE'): -285.118579955482,
        ('ASC_SR2', 'ASC_SR2'): 6156.860848428152,
        ('ASC_SR2', 'ASC_SR3P'): -412.00155512537197,
        ('ASC_SR2', 'ASC_TRAN'): -1312.7312255613883,
        ('ASC_SR2', 'ASC_WALK'): -461.36148965146094,
        ('ASC_SR2', 'MU_car'): 3491.370489213462,
        ('ASC_SR2', 'MU_nonmotor'): -244.85789490150424,
        ('ASC_SR2', 'hhinc#2'): 360960.5064929566,
        ('ASC_SR2', 'hhinc#3'): -25836.407171358624,
        ('ASC_SR2', 'hhinc#4'): -73800.95798232155,
        ('ASC_SR2', 'hhinc#5'): -16063.872652571805,
        ('ASC_SR2', 'hhinc#6'): -23821.208247582188,
        ('ASC_SR2', 'totcost'): -255314.7240133045,
        ('ASC_SR2', 'tottime'): -26043.721750561366,
        ('ASC_SR3P', 'ASC_BIKE'): 19.760852749005203,
        ('ASC_SR3P', 'ASC_SR2'): -412.00155512537197,
        ('ASC_SR3P', 'ASC_SR3P'): 194.01812799791654,
        ('ASC_SR3P', 'ASC_TRAN'): 75.72048947864153,
        ('ASC_SR3P', 'ASC_WALK'): 21.26889182584585,
        ('ASC_SR3P', 'MU_car'): 108.36680636408329,
        ('ASC_SR3P', 'MU_nonmotor'): 11.214695666269147,
        ('ASC_SR3P', 'hhinc#2'): -25836.407171358624,
        ('ASC_SR3P', 'hhinc#3'): 11951.689201021742,
        ('ASC_SR3P', 'hhinc#4'): 4654.093182472214,
        ('ASC_SR3P', 'hhinc#5'): 1231.298948075397,
        ('ASC_SR3P', 'hhinc#6'): 1291.5421535124285,
        ('ASC_SR3P', 'totcost'): -873.8405986324798,
        ('ASC_SR3P', 'tottime'): 2847.414853230762,
        ('ASC_TRAN', 'ASC_BIKE'): 76.573987238185,
        ('ASC_TRAN', 'ASC_SR2'): -1312.7312255613883,
        ('ASC_TRAN', 'ASC_SR3P'): 75.72048947864153,
        ('ASC_TRAN', 'ASC_TRAN'): 795.8802149505714,
        ('ASC_TRAN', 'ASC_WALK'): 106.98212315599287,
        ('ASC_TRAN', 'MU_car'): -930.0194408568389,
        ('ASC_TRAN', 'MU_nonmotor'): 58.620406514681704,
        ('ASC_TRAN', 'hhinc#2'): -73800.95798232155,
        ('ASC_TRAN', 'hhinc#3'): 4654.093182472214,
        ('ASC_TRAN', 'hhinc#4'): 43744.97724907036,
        ('ASC_TRAN', 'hhinc#5'): 4436.148875125189,
        ('ASC_TRAN', 'hhinc#6'): 5766.003794801971,
        ('ASC_TRAN', 'totcost'): -289.26636202646205,
        ('ASC_TRAN', 'tottime'): 17843.995404571957,
        ('ASC_WALK', 'ASC_BIKE'): 32.348453110164314,
        ('ASC_WALK', 'ASC_SR2'): -461.36148965146094,
        ('ASC_WALK', 'ASC_SR3P'): 21.26889182584585,
        ('ASC_WALK', 'ASC_TRAN'): 106.98212315599287,
        ('ASC_WALK', 'ASC_WALK'): 257.4268216790701,
        ('ASC_WALK', 'MU_car'): -372.1302408319178,
        ('ASC_WALK', 'MU_nonmotor'): 84.61754029339109,
        ('ASC_WALK', 'hhinc#2'): -23821.208247582188,
        ('ASC_WALK', 'hhinc#3'): 1291.5421535124283,
        ('ASC_WALK', 'hhinc#4'): 5766.003794801973,
        ('ASC_WALK', 'hhinc#5'): 1870.2419763938155,
        ('ASC_WALK', 'hhinc#6'): 12476.241347103962,
        ('ASC_WALK', 'totcost'): -4472.049603002317,
        ('ASC_WALK', 'tottime'): 7147.124392574635,
        ('MU_car', 'ASC_BIKE'): -219.27982884201612,
        ('MU_car', 'ASC_SR2'): 3491.370489213462,
        ('MU_car', 'ASC_SR3P'): 108.36680636408329,
        ('MU_car', 'ASC_TRAN'): -930.0194408568389,
        ('MU_car', 'ASC_WALK'): -372.1302408319178,
        ('MU_car', 'MU_car'): 3048.745424759559,
        ('MU_car', 'MU_nonmotor'): -215.09078801898488,
        ('MU_car', 'hhinc#2'): 209343.58472860648,
        ('MU_car', 'hhinc#3'): 4904.0435782757395,
        ('MU_car', 'hhinc#4'): -54645.9131974324,
        ('MU_car', 'hhinc#5'): -12553.188578813708,
        ('MU_car', 'hhinc#6'): -20397.013767114706,
        ('MU_car', 'totcost'): -130926.84443581512,
        ('MU_car', 'tottime'): -19903.25282015786,
        ('MU_nonmotor', 'ASC_BIKE'): 69.38016083689357,
        ('MU_nonmotor', 'ASC_SR2'): -244.85789490150424,
        ('MU_nonmotor', 'ASC_SR3P'): 11.214695666269147,
        ('MU_nonmotor', 'ASC_TRAN'): 58.620406514681704,
        ('MU_nonmotor', 'ASC_WALK'): 84.61754029339109,
        ('MU_nonmotor', 'MU_car'): -215.09078801898488,
        ('MU_nonmotor', 'MU_nonmotor'): 106.11047953795583,
        ('MU_nonmotor', 'hhinc#2'): -13788.1097937347,
        ('MU_nonmotor', 'hhinc#3'): 711.5550773867994,
        ('MU_nonmotor', 'hhinc#4'): 3435.2281478513137,
        ('MU_nonmotor', 'hhinc#5'): 3589.230542100178,
        ('MU_nonmotor', 'hhinc#6'): 4773.341396211997,
        ('MU_nonmotor', 'totcost'): -571.9687206358277,
        ('MU_nonmotor', 'tottime'): 3082.561197668304,
        ('hhinc#2', 'ASC_BIKE'): -16063.872652571805,
        ('hhinc#2', 'ASC_SR2'): 360960.5064929566,
        ('hhinc#2', 'ASC_SR3P'): -25836.407171358624,
        ('hhinc#2', 'ASC_TRAN'): -73800.95798232155,
        ('hhinc#2', 'ASC_WALK'): -23821.208247582188,
        ('hhinc#2', 'MU_car'): 209343.58472860648,
        ('hhinc#2', 'MU_nonmotor'): -13788.1097937347,
        ('hhinc#2', 'hhinc#2'): 27739015.863625906,
        ('hhinc#2', 'hhinc#3'): -2107887.124567991,
        ('hhinc#2', 'hhinc#4'): -5551797.986970259,
        ('hhinc#2', 'hhinc#5'): -1180261.9541857818,
        ('hhinc#2', 'hhinc#6'): -1731206.5786703678,
        ('hhinc#2', 'totcost'): -15915701.570008647,
        ('hhinc#2', 'tottime'): -1404099.9397647784,
        ('hhinc#3', 'ASC_BIKE'): 1231.2989480753972,
        ('hhinc#3', 'ASC_SR2'): -25836.407171358624,
        ('hhinc#3', 'ASC_SR3P'): 11951.689201021742,
        ('hhinc#3', 'ASC_TRAN'): 4654.093182472214,
        ('hhinc#3', 'ASC_WALK'): 1291.5421535124283,
        ('hhinc#3', 'MU_car'): 4904.0435782757395,
        ('hhinc#3', 'MU_nonmotor'): 711.5550773867994,
        ('hhinc#3', 'hhinc#2'): -2107887.124567991,
        ('hhinc#3', 'hhinc#3'): 985365.3310746389,
        ('hhinc#3', 'hhinc#4'): 365082.18492341647,
        ('hhinc#3', 'hhinc#5'): 96863.55545307972,
        ('hhinc#3', 'hhinc#6'): 105646.64587551646,
        ('hhinc#3', 'totcost'): -18197.96776085889,
        ('hhinc#3', 'tottime'): 173393.88742844868,
        ('hhinc#4', 'ASC_BIKE'): 4436.148875125189,
        ('hhinc#4', 'ASC_SR2'): -73800.95798232155,
        ('hhinc#4', 'ASC_SR3P'): 4654.093182472214,
        ('hhinc#4', 'ASC_TRAN'): 43744.97724907036,
        ('hhinc#4', 'ASC_WALK'): 5766.003794801973,
        ('hhinc#4', 'MU_car'): -54645.9131974324,
        ('hhinc#4', 'MU_nonmotor'): 3435.2281478513137,
        ('hhinc#4', 'hhinc#2'): -5551797.986970259,
        ('hhinc#4', 'hhinc#3'): 365082.18492341647,
        ('hhinc#4', 'hhinc#4'): 3238425.3752979534,
        ('hhinc#4', 'hhinc#5'): 328415.284903046,
        ('hhinc#4', 'hhinc#6'): 431510.21040576114,
        ('hhinc#4', 'totcost'): -203942.9008935652,
        ('hhinc#4', 'tottime'): 996131.649172921,
        ('hhinc#5', 'ASC_BIKE'): 5322.601183863066,
        ('hhinc#5', 'ASC_SR2'): -16063.872652571805,
        ('hhinc#5', 'ASC_SR3P'): 1231.298948075397,
        ('hhinc#5', 'ASC_TRAN'): 4436.148875125189,
        ('hhinc#5', 'ASC_WALK'): 1870.2419763938155,
        ('hhinc#5', 'MU_car'): -12553.188578813708,
        ('hhinc#5', 'MU_nonmotor'): 3589.230542100178,
        ('hhinc#5', 'hhinc#2'): -1180261.9541857818,
        ('hhinc#5', 'hhinc#3'): 96863.55545307972,
        ('hhinc#5', 'hhinc#4'): 328415.284903046,
        ('hhinc#5', 'hhinc#5'): 376816.62222404545,
        ('hhinc#5', 'hhinc#6'): 139543.8152706944,
        ('hhinc#5', 'totcost'): 64871.45953365474,
        ('hhinc#5', 'tottime'): 191184.20323081187,
        ('hhinc#6', 'ASC_BIKE'): 1870.2419763938155,
        ('hhinc#6', 'ASC_SR2'): -23821.208247582188,
        ('hhinc#6', 'ASC_SR3P'): 1291.5421535124285,
        ('hhinc#6', 'ASC_TRAN'): 5766.003794801971,
        ('hhinc#6', 'ASC_WALK'): 12476.241347103962,
        ('hhinc#6', 'MU_car'): -20397.013767114706,
        ('hhinc#6', 'MU_nonmotor'): 4773.341396211997,
        ('hhinc#6', 'hhinc#2'): -1731206.5786703678,
        ('hhinc#6', 'hhinc#3'): 105646.64587551646,
        ('hhinc#6', 'hhinc#4'): 431510.21040576114,
        ('hhinc#6', 'hhinc#5'): 139543.8152706944,
        ('hhinc#6', 'hhinc#6'): 872604.6192807175,
        ('hhinc#6', 'totcost'): -95081.8079551341,
        ('hhinc#6', 'tottime'): 364196.75026433234,
        ('totcost', 'ASC_BIKE'): 651.7010091466225,
        ('totcost', 'ASC_SR2'): -255314.7240133045,
        ('totcost', 'ASC_SR3P'): -873.8405986324798,
        ('totcost', 'ASC_TRAN'): -289.26636202646205,
        ('totcost', 'ASC_WALK'): -4472.049603002317,
        ('totcost', 'MU_car'): -130926.84443581512,
        ('totcost', 'MU_nonmotor'): -571.9687206358277,
        ('totcost', 'hhinc#2'): -15915701.570008647,
        ('totcost', 'hhinc#3'): -18197.96776085889,
        ('totcost', 'hhinc#4'): -203942.9008935652,
        ('totcost', 'hhinc#5'): 64871.45953365474,
        ('totcost', 'hhinc#6'): -95081.8079551341,
        ('totcost', 'totcost'): 67516567.00440338,
        ('totcost', 'tottime'): -775245.3645765011,
        ('tottime', 'ASC_BIKE'): 3410.093422258546,
        ('tottime', 'ASC_SR2'): -26043.721750561366,
        ('tottime', 'ASC_SR3P'): 2847.414853230762,
        ('tottime', 'ASC_TRAN'): 17843.995404571957,
        ('tottime', 'ASC_WALK'): 7147.124392574635,
        ('tottime', 'MU_car'): -19903.25282015786,
        ('tottime', 'MU_nonmotor'): 3082.561197668304,
        ('tottime', 'hhinc#2'): -1404099.9397647784,
        ('tottime', 'hhinc#3'): 173393.88742844868,
        ('tottime', 'hhinc#4'): 996131.649172921,
        ('tottime', 'hhinc#5'): 191184.20323081187,
        ('tottime', 'hhinc#6'): 364196.75026433234,
        ('tottime', 'totcost'): -775245.3645765011,
        ('tottime', 'tottime'): 910724.9012464393,
    }
    assert dict(ll1.bhhh.unstack()) == approx(bhhh_correct)
    assert dict(ll2.bhhh.unstack()) == approx(bhhh_correct)

    dll_casewise_A = ll2.dll_casewise / j2.weight_normalization
    dll_casewise_B = np.asarray(dll_casewise_A) / j2.data_wt.values

    corrected_bhhh = pd.DataFrame(
        np.dot(dll_casewise_A.T, dll_casewise_B),
        index=ll2.dll.index,
        columns=ll2.dll.index,
    ) * j2.weight_normalization

    assert dict(ll1.bhhh.unstack()) == approx(dict(corrected_bhhh.unstack()))
