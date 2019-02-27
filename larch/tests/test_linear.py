
from larch.model.linear import ParameterRef_C, DataRef_C, LinearComponent_C, LinearFunction_C
from larch import P, X, PX
import keyword
import pytest

def test_parameter_c():

	p = ParameterRef_C("hsh")
	assert "hsh" == p
	assert p == "hsh"
	assert not keyword.iskeyword(p)
	assert hash(p) == hash("hsh")
	assert repr(p) == "P.hsh"
	assert p == P.hsh
	assert p == P("hsh")
	assert p == P['hsh']

def test_data_c():

	d = DataRef_C("hsh")
	assert "hsh" == d
	assert d == "hsh"
	assert not keyword.iskeyword(d)
	assert hash(d) == hash("hsh")
	assert repr(d) == "X.hsh"

	assert d == X.hsh
	assert d == X("hsh")
	assert d == X['hsh']

	p = ParameterRef_C("hsh")
	assert not p == d
	assert p != d

def test_data_c_math():

	assert X.Aaa + X.Bbb == X("Aaa+Bbb")
	assert X.Aaa - X.Bbb == X("Aaa-Bbb")
	assert X.Aaa * X.Bbb == X("Aaa*Bbb")
	assert X.Aaa / X.Bbb == X("Aaa/Bbb")
	assert X.Aaa & X.Bbb == X("Aaa&Bbb")
	assert X.Aaa | X.Bbb == X("Aaa|Bbb")
	assert X.Aaa ^ X.Bbb == X("Aaa^Bbb")
	assert X.Aaa ** X.Bbb == X("Aaa**Bbb")
	assert X.Zzz / X.Aaa + X.Vvv * X.Bbb == X('(Zzz/Aaa)+(Vvv*Bbb)')
	assert +X.Aaa == X("Aaa")
	assert -X.Aaa == X("-Aaa")

	assert X.Aaa + 2 == X("Aaa+2")
	assert X.Aaa - 2 == X("Aaa-2")
	assert X.Aaa * 2 == X("Aaa*2")
	assert X.Aaa / 2 == X("Aaa/2")
	assert X.Aaa & 2 == X("Aaa&2")
	assert X.Aaa | 2 == X("Aaa|2")
	assert X.Aaa ^ 2 == X("Aaa^2")
	assert X.Aaa ** 2 == X("Aaa**2")

	assert 2 + X.Aaa == X("2+Aaa")
	assert 2 - X.Aaa == X("2-Aaa")
	assert 2 * X.Aaa == X("2*Aaa")
	assert 2 / X.Aaa == X("2/Aaa")
	assert 2 & X.Aaa == X("2&Aaa")
	assert 2 | X.Aaa == X("2|Aaa")
	assert 2 ^ X.Aaa == X("2^Aaa")
	assert 2 ** X.Aaa == X("2**Aaa")

	assert X.Aaa + 0 == X.Aaa
	assert 0 + X.Aaa == X.Aaa

	assert X.Aaa * 1 == X.Aaa
	assert 1 * X.Aaa == X.Aaa

	with pytest.raises(NotImplementedError):
		_ = X.Aaa + "Plain String"

	with pytest.raises(NotImplementedError):
		_ = X.Aaa - "Plain String"

	with pytest.raises(NotImplementedError):
		_ = X.Aaa * "Plain String"

	with pytest.raises(NotImplementedError):
		_ = X.Aaa / "Plain String"


def test_ref_gen():

	assert X["Asd"] == X("Asd") == X.Asd
	assert P["Asd"] == P("Asd") == P.Asd
	assert X.Asd != P.Asd

def test_linear_func():

	assert LinearComponent_C(param="pname", data="dname") == P.pname * X.dname

	assert type(list(P.singleton + P.pname * X.dname)[0]) is LinearComponent_C
	assert type(list(P.singleton + P.pname * X.dname)[1]) is LinearComponent_C

	assert type(list( + P.pname * X.dname + P.singleton)[0]) is LinearComponent_C
	assert type(list( + P.pname * X.dname + P.singleton)[1]) is LinearComponent_C



	assert list(-(P.pname * X.dname + P.singleton)) == [
		LinearComponent_C('pname', 'dname', -1.0),
		LinearComponent_C('singleton', '1', -1.0),
	]
	assert list(-(P.pname * X.dname - P.singleton)) == [
		LinearComponent_C('pname', 'dname', -1.0),
		LinearComponent_C('singleton', '1', 1.0),
	]

	assert list((P.pname * X.dname - P.singleton) * X.Sss) == [
		LinearComponent_C(param='pname', data='dname*Sss', scale=1.0),
		LinearComponent_C(param='singleton', data='Sss', scale=-1.0),
	]

	assert list(sum(PX(i) for i in ['Aaa', 'Bbb'])) == [
		LinearComponent_C(param='Aaa', data='Aaa', scale=1.0),
		LinearComponent_C(param='Bbb', data='Bbb', scale=1.0),
	]
