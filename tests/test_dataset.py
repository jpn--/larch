import larch.numba as lx
import pytest
from pytest import approx

def test_dataset():
    d = lx.examples.MTC(format='dataset')

    assert d.CASEID == 'caseid'
    assert d.ALTID == 'altid'

    assert d.chose.CASEID == 'caseid'
    assert d.chose.ALTID == 'altid'

    assert d['chose'].CASEID == 'caseid'
    assert d['chose'].ALTID == 'altid'
