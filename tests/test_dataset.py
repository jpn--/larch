import larch.numba as lx
import pytest
from pytest import approx

def test_dataset():
    d = lx.examples.MTC(format='dataset')

    assert d.flow.CASEID == 'caseid'
    assert d.flow.ALTID == 'altid'

    assert d.flow.chose.flow.CASEID == 'caseid'
    assert d.flow.chose.flow.ALTID == 'altid'

    assert d.flow['chose'].flow.CASEID == 'caseid'
    assert d.flow['chose'].flow.ALTID == 'altid'
