
from larch.model.linear import ParameterRef_C, DataRef_C, P, X
import keyword


def test_parameter_c():

	p = ParameterRef_C("hsh")
	assert "hsh" == p
	assert p == "hsh"
	assert not keyword.iskeyword(p)
	assert hash(p) == hash("hsh")
	assert repr(p) == "P.hsh"


def test_data_c():

	d = DataRef_C("hsh")
	assert "hsh" == d
	assert d == "hsh"
	assert not keyword.iskeyword(d)
	assert hash(d) == hash("hsh")
	assert repr(d) == "X.hsh"

	p = ParameterRef_C("hsh")
	assert not p == d
	assert p != d

def test_ref_gen():

	assert X["Asd"] == X("Asd") == X.Asd
	assert P["Asd"] == P("Asd") == P.Asd
	assert X.Asd != P.Asd
