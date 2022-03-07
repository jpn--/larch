import larch.numba as lx
import pytest
from pytest import approx

def test_dataset():
    d = lx.examples.MTC(format='dataset')

    assert d.dc.CASEID == 'caseid'
    assert d.dc.ALTID == 'altid'

    assert d.dc.chose.dc.CASEID == 'caseid'
    assert d.dc.chose.dc.ALTID == 'altid'

    assert d.dc['chose'].dc.CASEID == 'caseid'
    assert d.dc['chose'].dc.ALTID == 'altid'
