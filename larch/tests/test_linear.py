
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


	u = P.Aaa * X.Aaa + P.Bbb * X.Bbb
	u += P.Ccc * X.Ccc

	assert u == P.Aaa * X.Aaa + P.Bbb * X.Bbb + P.Ccc * X.Ccc

def test_piecewise_linear():
	from larch.util.data_expansion import piecewise_linear

	func = piecewise_linear(X.DataName, P.ParamName, [3, 5, 7])
	assert func[0] == P('ParamName ① up to 3') * X('piece(DataName,None,3)')
	assert func[1] == P('ParamName ② 3 to 5') * X('piece(DataName,3,5)')
	assert func[2] == P('ParamName ③ 5 to 7') * X('piece(DataName,5,7)')
	assert func[3] == P('ParamName ④ over 7') * X('piece(DataName,7,None)')
	assert len(func) == 4

	func = piecewise_linear(X.DataName, breaks=[3, 5, 7])
	assert func[0] == P('DataName ① up to 3') * X('piece(DataName,None,3)')
	assert func[1] == P('DataName ② 3 to 5') * X('piece(DataName,3,5)')
	assert func[2] == P('DataName ③ 5 to 7') * X('piece(DataName,5,7)')
	assert func[3] == P('DataName ④ over 7') * X('piece(DataName,7,None)')
	assert len(func) == 4

	func = piecewise_linear('GenName', breaks=[3, 5, 7])
	assert func[0] == P('GenName ① up to 3') * X('piece(GenName,None,3)')
	assert func[1] == P('GenName ② 3 to 5') * X('piece(GenName,3,5)')
	assert func[2] == P('GenName ③ 5 to 7') * X('piece(GenName,5,7)')
	assert func[3] == P('GenName ④ over 7') * X('piece(GenName,7,None)')
	assert len(func) == 4

	with pytest.raises(ValueError):
		func = piecewise_linear('GenName', [3, 5, 7])


