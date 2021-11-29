from pytest import approx, importorskip
import numpy as np
sh = importorskip("sharrow")
import larch.numba as lx
import larch.numba.data_arrays as lxd


def test_weighted():
    m = lx.example(1)
    m.choice_ca_var = 'chose'
    m.weight_co_var = 'hhinc+100'
    ds = lxd.to_dataset(m.dataservice)
    y, flows = lxd.prepare_data(ds, m)
    assert isinstance(y, sh.Dataset)
    assert list(y.coords.keys()) == ['_0_caseid_', 'alt_names', '_1_altid_', 'var_co', 'var_ca']
    assert list(y.keys()) == ['co', 'ca', 'ch', 'wt', 'av']
    assert y.dims == {'_0_caseid_': 5029, '_1_altid_': 6, 'var_co': 1, 'var_ca': 2}
    assert y.wt.values[:3] == approx(np.array([142.5, 117.5, 112.5], dtype=np.float32))


def test_choice_code():
    m = lx.example(1)
    m.choice_co_code = 'chosen_alt'
    m.weight_co_var = 'hhinc+100'
    chosen_alt = np.where(m.dataservice.data_ca['chose'].unstack())[1] + 1
    ds = lxd.to_dataset(m.dataservice)
    ds['chosen_alt'] = sh.DataArray(
        chosen_alt,
        dims=['_0_caseid_',],
    )
    y, flows = lxd.prepare_data(ds, m)
    assert isinstance(y, sh.Dataset)
    assert list(y.coords.keys()) == ['_0_caseid_', 'alt_names', '_1_altid_', '_var_co', '_var_ca']
    assert list(y.keys()) == ['co', 'ca', 'ch', 'wt', 'av']
    assert y.dims == {'_0_caseid_': 5029, '_1_altid_': 6, '_var_co': 1, '_var_ca': 2}
    assert y.wt.values[:3] == approx(np.array([142.5, 117.5, 112.5], dtype=np.float32))


def test_shared_data():
    m = lx.example(1)
    m.choice_ca_var = 'chose'
    m.weight_co_var = 'hhinc+100'
    ds = lxd.to_dataset(m.dataservice)
    pool = lx.DataTree(ds)
    y, flows = lxd.prepare_data(pool, m)
    assert isinstance(y, sh.Dataset)
    assert list(y.coords.keys()) == ['_0_caseid_', 'alt_names', '_1_altid_', 'var_co', 'var_ca']
    assert list(y.keys()) == ['co', 'ca', 'ch', 'wt', 'av']
    assert y.dims == {'_0_caseid_': 5029, '_1_altid_': 6, 'var_co': 1, 'var_ca': 2}
    assert y.wt.values[:3] == approx(np.array([142.5, 117.5, 112.5], dtype=np.float32))


