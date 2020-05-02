import larch
import larch.model.linear_math
from larch.model.linear import ParameterRef_C, DataRef_C, LinearComponent_C, LinearFunction_C, _null_
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
	assert X.Aaa * 2 == LinearComponent_C(param=_null_, scale=2, data='Aaa')
	assert X.Aaa / 2 == X("Aaa/2")
	assert X.Aaa & 2 == X("Aaa&2")
	assert X.Aaa | 2 == X("Aaa|2")
	assert X.Aaa ^ 2 == X("Aaa^2")
	assert X.Aaa ** 2 == X("Aaa**2")

	assert 2 + X.Aaa == X("2+Aaa")
	assert 2 - X.Aaa == X("2-Aaa")
	assert 2 * X.Aaa == LinearComponent_C(param=_null_, scale=2, data='Aaa')
	assert 2 / X.Aaa == X("2/Aaa")
	assert 2 & X.Aaa == X("2&Aaa")
	assert 2 | X.Aaa == X("2|Aaa")
	assert 2 ^ X.Aaa == X("2^Aaa")
	assert 2 ** X.Aaa == X("2**Aaa")

	assert X.Aaa + 0 == X.Aaa
	assert 0 + X.Aaa == X.Aaa

	assert X.Aaa * 1 == X.Aaa
	assert 1 * X.Aaa == X.Aaa

	with pytest.raises(TypeError):
		_ = X.Aaa + "Plain String"

	with pytest.raises(TypeError):
		_ = X.Aaa - "Plain String"

	with pytest.raises(TypeError):
		_ = X.Aaa * "Plain String"

	with pytest.raises(TypeError):
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

	assert P.ppp * X.xxx * 1.234 == P.ppp * 1.234 * X.xxx
	assert P.ppp * X.xxx * 1.234 == X.xxx * P.ppp * 1.234
	assert P.ppp * X.xxx * 1.234 == X.xxx * 1.234 * P.ppp
	assert P.ppp * X.xxx * 1.234 == 1.234 * X.xxx * P.ppp
	assert P.ppp * X.xxx * 1.234 == 1.234 * P.ppp * X.xxx

	assert (P.ppp * X.xxx) * 1.234 == P.ppp * (1.234 * X.xxx)
	assert (P.ppp * X.xxx) * 1.234 == X.xxx * (P.ppp * 1.234)
	assert (P.ppp * X.xxx) * 1.234 == X.xxx * (1.234 * P.ppp)
	assert (P.ppp * X.xxx) * 1.234 == 1.234 * (X.xxx * P.ppp)
	assert (P.ppp * X.xxx) * 1.234 == 1.234 * (P.ppp * X.xxx)

	assert (P.ppp * X.xxx) * 1.234 == (P.ppp * 1.234) * X.xxx
	assert (P.ppp * X.xxx) * 1.234 == (X.xxx * P.ppp) * 1.234
	assert (P.ppp * X.xxx) * 1.234 == (X.xxx * 1.234) * P.ppp
	assert (P.ppp * X.xxx) * 1.234 == (1.234 * X.xxx) * P.ppp
	assert (P.ppp * X.xxx) * 1.234 == (1.234 * P.ppp) * X.xxx

	assert (P.ppp * X.xxx * 1.234) == P.ppp * (1.234 * X.xxx)
	assert (P.ppp * X.xxx * 1.234) == X.xxx * (P.ppp * 1.234)
	assert (P.ppp * X.xxx * 1.234) == X.xxx * (1.234 * P.ppp)
	assert (P.ppp * X.xxx * 1.234) == 1.234 * (X.xxx * P.ppp)
	assert (P.ppp * X.xxx * 1.234) == 1.234 * (P.ppp * X.xxx)

	assert (P.ppp * X.xxx * 1.234) == (P.ppp * 1.234) * X.xxx
	assert (P.ppp * X.xxx * 1.234) == (X.xxx * P.ppp) * 1.234
	assert (P.ppp * X.xxx * 1.234) == (X.xxx * 1.234) * P.ppp
	assert (P.ppp * X.xxx * 1.234) == (1.234 * X.xxx) * P.ppp
	assert (P.ppp * X.xxx * 1.234) == (1.234 * P.ppp) * X.xxx

	assert P.ppp * (X.xxx * 1.234) == P.ppp * (1.234 * X.xxx)
	assert P.ppp * (X.xxx * 1.234) == X.xxx * (P.ppp * 1.234)
	assert P.ppp * (X.xxx * 1.234) == X.xxx * (1.234 * P.ppp)
	assert P.ppp * (X.xxx * 1.234) == 1.234 * (X.xxx * P.ppp)
	assert P.ppp * (X.xxx * 1.234) == 1.234 * (P.ppp * X.xxx)

	assert P.ppp * (X.xxx * 1.234) == (P.ppp * 1.234) * X.xxx
	assert P.ppp * (X.xxx * 1.234) == (X.xxx * P.ppp) * 1.234
	assert P.ppp * (X.xxx * 1.234) == (X.xxx * 1.234) * P.ppp
	assert P.ppp * (X.xxx * 1.234) == (1.234 * X.xxx) * P.ppp
	assert P.ppp * (X.xxx * 1.234) == (1.234 * P.ppp) * X.xxx

	assert (P.ppp * X.xxx) * X.xxx == P.ppp * X('xxx*xxx')
	assert (P.ppp * X.xxx) * (P("_") * X.xxx) == P.ppp * X('xxx*xxx')
	assert (P("_") * X.xxx) * (P.ppp * X.xxx) == P.ppp * X('xxx*xxx')

	# Test squaring a boolean
	assert (P.ppp * X('boolean(xxx)')) * X('boolean(xxx)') == P.ppp * X('boolean(xxx)')
	assert (P.ppp * X('boolean(xxx)')) * (P("_") * X('boolean(xxx)')) == P.ppp * X('boolean(xxx)')

	assert ((P.p1 * X.x1 + P.p2 * X.x2) * (P('_') * 1.1 * X.x1 + P('_') * 2 * X.x2)) == (
			P.p1 * 1.1 * X('x1*x1') + P.p1 * 2.0 * X('x1*x2') + P.p2 * 1.1 * X('x2*x1') + P.p2 * 2.0 * X('x2*x2')
	)

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


def test_piecewise_expansion():
	import pandas, io, numpy
	from larch.util.data_expansion import piecewise_expansion, piecewise_linear

	df = pandas.DataFrame(
		numpy.linspace(0, 10, 25),
		columns=['DataName'],
	)

	expanded = piecewise_expansion(df, [3, 5, 7])

	s = '''
	    piece(DataName,None,3)  piece(DataName,3,5)  piece(DataName,5,7)  piece(DataName,7,None)
	0                 0.000000             0.000000             0.000000                0.000000
	1                 0.416667             0.000000             0.000000                0.000000
	2                 0.833333             0.000000             0.000000                0.000000
	3                 1.250000             0.000000             0.000000                0.000000
	4                 1.666667             0.000000             0.000000                0.000000
	5                 2.083333             0.000000             0.000000                0.000000
	6                 2.500000             0.000000             0.000000                0.000000
	7                 2.916667             0.000000             0.000000                0.000000
	8                 3.000000             0.333333             0.000000                0.000000
	9                 3.000000             0.750000             0.000000                0.000000
	10                3.000000             1.166667             0.000000                0.000000
	11                3.000000             1.583333             0.000000                0.000000
	12                3.000000             2.000000             0.000000                0.000000
	13                3.000000             2.000000             0.416667                0.000000
	14                3.000000             2.000000             0.833333                0.000000
	15                3.000000             2.000000             1.250000                0.000000
	16                3.000000             2.000000             1.666667                0.000000
	17                3.000000             2.000000             2.000000                0.083333
	18                3.000000             2.000000             2.000000                0.500000
	19                3.000000             2.000000             2.000000                0.916667
	20                3.000000             2.000000             2.000000                1.333333
	21                3.000000             2.000000             2.000000                1.750000
	22                3.000000             2.000000             2.000000                2.166667
	23                3.000000             2.000000             2.000000                2.583333
	24                3.000000             2.000000             2.000000                3.000000
	'''

	correct = pandas.read_csv(io.StringIO(s), sep='\s+')

	pandas.testing.assert_frame_equal(expanded, correct)

	func = piecewise_linear(X.DataName, P.ParamName, [3, 5, 7])
	expanded2 = piecewise_expansion(df, func)
	pandas.testing.assert_frame_equal(expanded2, correct)


def test_pmath():
	m = larch.Model()
	m.utility_ca = P.Aaa * X.Aaa + P.Bbb * X.Bbb + P.Ccc
	m.set_values(Aaa=12, Bbb=20, Ccc=2)

	assert P.Aaa.value(m) == 12
	assert P.Bbb.value(m) == 20
	assert P.Ccc.value(m) == 2

	with pytest.raises(KeyError):
		P.Ddd.value(m)

	assert P.Ddd.value(m, {'Ddd': 123}) == 123

	y = P.Aaa + P.Bbb
	assert y.value(m) == 12 + 20
	assert repr(y) == 'P.Aaa + P.Bbb'
	assert repr(y + y) == 'P.Aaa + P.Bbb + P.Aaa + P.Bbb'
	assert isinstance(y, larch.model.linear.LinearFunction_C)
	assert isinstance(y + y, larch.model.linear.LinearFunction_C)

	y = P.Aaa / P.Bbb
	assert y.value(m) == 12 / 20
	assert y.string(m) == "0.6"
	assert repr(y) == 'P.Aaa / P.Bbb'
	assert repr(y + y) == 'P.Aaa / P.Bbb + P.Aaa / P.Bbb'

	y = P.Aaa * 60 / (P.Bbb * 100)
	y.set_fmt("${:0.2f}/hr")
	assert y.value(m) == 12 * 60 / (20 * 100)
	assert y.string(m) == '$0.36/hr'

	y = P.Aaa + P.Bbb / P.Ccc
	assert y.value(m) == 12 + 20 / 2
	assert isinstance(y, larch.model.linear_math.ParameterAdd)
	assert repr(y) == 'P.Aaa + P.Bbb / P.Ccc'

	y = (P.Aaa + P.Bbb) + P.Ccc * P.Aaa
	assert isinstance(y, larch.model.linear_math.ParameterAdd)
	assert y.value(m) == (12 + 20) + 2 * 12

	y = (P.Aaa + P.Bbb) + P.Ccc / P.Aaa
	assert isinstance(y, larch.model.linear_math.ParameterAdd)
	assert y.value(m) == (12 + 20) + 2 / 12

	y = (P.Aaa + P.Bbb + P.Ccc) * P.Aaa
	assert isinstance(y, larch.model.linear_math.ParameterMultiply)
	assert y.value(m) == (12 + 20 + 2) * 12

	y = (P.Aaa + P.Bbb + P.Ccc) / P.Aaa
	assert isinstance(y, larch.model.linear_math.ParameterDivide)
	assert y.value(m) == (12 + 20 + 2) / 12
	assert repr(y) == '(P.Aaa + P.Bbb + P.Ccc) / P.Aaa'

	y = (P.Aaa + P.Bbb + P.Ccc) - P.Aaa
	assert isinstance(y, larch.model.linear.LinearFunction_C)
	assert y.value(m) == 12 + 20 + 2 - 12


def test_pvalue():
	m = larch.Model()
	m.utility_ca = P.Aaa * X.Aaa + P.Bbb * X.Bbb + P.Ccc
	m.set_values(Aaa=12, Bbb=20, Ccc=2)

	assert P.Aaa.value(m) == 12
	assert P.Bbb.value(m) == 20
	assert P.Ccc.value(m) == 2

	with pytest.raises(KeyError):
		P.Ddd.value(m)

	assert P.Ddd.value(m, {'Ddd': 123}) == 123

	y = P.Ddd / P.Bbb
	with pytest.raises(KeyError):
		m.pvalue(y, log_errors=False)
	assert m.pformat(y) == 'NA'

	y = P.Aaa + P.Bbb
	assert m.pvalue(y) == 12 + 20
	assert m.pformat(y) == str(12 + 20)

	y = P.Aaa / P.Bbb
	assert m.pvalue(y) == 12 / 20
	assert m.pformat(y) == "0.6"

	y = P.Aaa * 60 / (P.Bbb * 100)
	y.set_fmt("${:0.2f}/hr")
	assert m.pvalue(y) == 12 * 60 / (20 * 100)
	assert m.pformat(y) == '$0.36/hr'

	y = P.Aaa + P.Bbb / P.Ccc
	assert m.pvalue(y) == 12 + 20 / 2
	assert m.pformat(y) == '22'

	y = (P.Aaa + P.Bbb) + P.Ccc * P.Aaa
	assert m.pvalue(y) == (12 + 20) + 2 * 12
	assert m.pformat(y) == "{:.3g}".format((12 + 20) + 2 * 12)

	y = (P.Aaa + P.Bbb) + P.Ccc / P.Aaa
	assert m.pvalue(y) == (12 + 20) + 2 / 12
	assert m.pformat(y) == "{:.3g}".format((12 + 20) + 2 / 12)

	y = (P.Aaa + P.Bbb + P.Ccc) * P.Aaa
	assert m.pvalue(y) == (12 + 20 + 2) * 12
	assert m.pformat(y) == "{:.3g}".format((12 + 20 + 2) * 12)

	y = (P.Aaa + P.Bbb + P.Ccc) / P.Aaa
	assert m.pvalue(y) == (12 + 20 + 2) / 12
	assert m.pformat(y) == "{:.3g}".format((12 + 20 + 2) / 12)

	y = (P.Aaa + P.Bbb + P.Ccc) - P.Aaa
	assert m.pvalue(y) == 12 + 20 + 2 - 12
	assert m.pformat(y) == "{:.3g}".format(12 + 20 + 2 - 12)

	y = (P.Aaa + P.Bbb * 2) + P.Ccc * P.Aaa
	assert m.pvalue(y) == (12 + 20 * 2) + 2 * 12
	assert m.pformat(y) == "{:.3g}".format((12 + 20 * 2) + 2 * 12)

	y = (P.Aaa + P.Bbb) + P.Ccc / P.Aaa * 3
	assert m.pvalue(y) == (12 + 20) + 2 / 12 * 3
	assert m.pformat(y) == "{:.3g}".format((12 + 20) + 2 / 12 * 3)

	y = (P.Aaa + P.Bbb + P.Ccc) * P.Aaa * 3
	assert m.pvalue(y) == (12 + 20 + 2) * 12 * 3
	assert m.pformat(y) == "{:.3g}".format((12 + 20 + 2) * 12 * 3)

	y = (P.Aaa + P.Bbb + P.Ccc) / P.Aaa * 3
	assert m.pvalue(y) == (12 + 20 + 2) / 12 * 3
	assert m.pformat(y) == "{:.3g}".format((12 + 20 + 2) / 12 * 3)

	y = (P.Aaa + P.Bbb + P.Ccc) - P.Aaa * 3
	assert m.pvalue(y) == 12 + 20 + 2 - 12 * 3
	assert m.pformat(y) == "{:.3g}".format(12 + 20 + 2 - 12 * 3)

	y = (11 + P.Aaa + P.Bbb * 2) + P.Ccc * P.Aaa
	assert m.pvalue(y) == (11 + 12 + 20 * 2) + 2 * 12
	assert m.pformat(y) == "{:.3g}".format((11 + 12 + 20 * 2) + 2 * 12)

	y = (11 + P.Aaa + P.Bbb) + P.Ccc / P.Aaa * 3
	assert m.pvalue(y) == (11 + 12 + 20) + 2 / 12 * 3
	assert m.pformat(y) == "{:.3g}".format((11 + 12 + 20) + 2 / 12 * 3)

	y = (11 + P.Aaa + P.Bbb + P.Ccc) * P.Aaa * 3
	assert m.pvalue(y) == (11 + 12 + 20 + 2) * 12 * 3
	assert m.pformat(y) == "{:.3g}".format((11 + 12 + 20 + 2) * 12 * 3)

	y = (11 + P.Aaa + P.Bbb + P.Ccc) / P.Aaa * 3
	assert m.pvalue(y) == (11 + 12 + 20 + 2) / 12 * 3
	assert m.pformat(y) == "{:.3g}".format((11 + 12 + 20 + 2) / 12 * 3)

	y = (11 + P.Aaa + P.Bbb + P.Ccc) - P.Aaa * 3
	assert m.pvalue(y) == 11 + 12 + 20 + 2 - 12 * 3
	assert m.pformat(y) == "{:.3g}".format(11 + 12 + 20 + 2 - 12 * 3)

	with pytest.raises(NotImplementedError):
		P.Aaa * 10 + P.Bbb / 10 * X.Yyy + 2

	y = P.Aaa * 10 + P.Bbb / 10 + 2
	assert m.pvalue(y) == 12 * 10 + 20 / 10 + 2
	assert m.pformat(y) == "{:.3g}".format(12 * 10 + 20 / 10 + 2)

	with pytest.raises(Exception):
		m.utility_ca = P.Aaa + P.Ccc + 5

	yd = {
		'Yxx': P.Aaa + P.Bbb / P.Ccc,
		'Yyy': P.Aaa + P.Bbb,
		'Yzz': P.Aaa * 60 / (P.Bbb * 100),
	}
	assert m.pvalue(yd) == {
		'Yxx': 12 + 20 / 2,
		'Yyy': 12 + 20,
		'Yzz': 12 * 60 / (20 * 100),
	}
	assert m.pformat(yd) == {
		'Yxx': "{:.3g}".format(12 + 20 / 2),
		'Yyy': "{:.3g}".format(12 + 20),
		'Yzz': "{:.3g}".format(12 * 60 / (20 * 100)),
	}


def test_pmath_in_utility():
	d = larch.examples.MTC()
	m0 = larch.Model(dataservice=d)

	m0.utility_co[2] = P("ASC_SR2") * 10 + P("hhinc#2") / 10 * X("hhinc")
	m0.utility_co[3] = P("ASC_SR3P") * 10 + P("hhinc#3") / 10 * X("hhinc")
	m0.utility_co[4] = P("ASC_TRAN") * 10 + P("hhinc#4") / 10 * X("hhinc")
	m0.utility_co[5] = P("ASC_BIKE") * 10 + P("hhinc#5") / 10 * X("hhinc")
	m0.utility_co[6] = P("ASC_WALK") * 10 + P("hhinc#6") / 10 * X("hhinc")

	m0.utility_ca = (
			+ P("nonmotorized_time") / 10. * X("(altnum>4) * tottime")
			+ P("motorized_ovtt") * 10 * X("(altnum <= 4) * ovtt")
			+ P("motorized_ivtt") * X("(altnum <= 4) * ivtt")
			+ PX("totcost")
	)
	m0.availability_var = '_avail_'
	m0.choice_ca_var = '_choice_'

	m1 = larch.Model(dataservice=d)

	m1.utility_co[2] = P("ASC_SR2") * X('10') + P("hhinc#2") * X("hhinc/10")
	m1.utility_co[3] = P("ASC_SR3P") * X('10') + P("hhinc#3") * X("hhinc/10")
	m1.utility_co[4] = P("ASC_TRAN") * X('10') + P("hhinc#4") * X("hhinc/10")
	m1.utility_co[5] = P("ASC_BIKE") * X('10') + P("hhinc#5") * X("hhinc/10")
	m1.utility_co[6] = P("ASC_WALK") * X('10') + P("hhinc#6") * X("hhinc/10")

	m1.utility_ca = (
			+ P("nonmotorized_time") * X("(altnum>4) * tottime / 10")
			+ P("motorized_ovtt") * X("(altnum <= 4) * ovtt * 10")
			+ P("motorized_ivtt") * X("(altnum <= 4) * ivtt")
			+ PX("totcost")
	)
	m1.availability_var = '_avail_'
	m1.choice_ca_var = '_choice_'

	m0.load_data()
	m1.load_data()

	r0 = m0.maximize_loglike(quiet=True)
	r1 = m1.maximize_loglike(quiet=True)
	assert r0.loglike == pytest.approx(-3587.6430040944942)
	assert r1.loglike == pytest.approx(-3587.6430040944942)

	m0.calculate_parameter_covariance()
	m1.calculate_parameter_covariance()
	t = {
		'ASC_BIKE': -5.318650574990901,
		'ASC_SR2': -22.291563439182628,
		'ASC_SR3P': -22.174552606750527,
		'ASC_TRAN': -3.293923857045225,
		'ASC_WALK': 1.6172450189610719,
		'hhinc#2': -1.4000897138949544,
		'hhinc#3': 0.12900984170888324,
		'hhinc#4': -3.0601742475362923,
		'hhinc#5': -2.333410249527477,
		'hhinc#6': -3.048442130390144,
		'motorized_ivtt': -0.4116740527068954,
		'motorized_ovtt': -12.958446214791113,
		'nonmotorized_time': -11.789244777056298,
		'totcost': -20.19350165272386,
	}
	assert dict(m0.pf['t_stat']) == pytest.approx(t, rel=1e-5)
	assert dict(m1.pf['t_stat']) == pytest.approx(t, rel=1e-5)

	assert (m0.get_value(P.motorized_ivtt) * 60) / (m0.get_value(P.totcost) * 100) == pytest.approx(0.3191492801963062)
	assert m0.get_value( (P.motorized_ivtt * 60) / (P.totcost * 100) ) == pytest.approx(0.3191492801963062)
	assert (m1.get_value(P.motorized_ivtt) * 60) / (m1.get_value(P.totcost) * 100) == pytest.approx(0.3191492801963062)
	assert m1.get_value( (P.motorized_ivtt * 60) / (P.totcost * 100) ) == pytest.approx(0.3191492801963062)

def test_linear_function_iadd():
	# Test inplace add on unattached LinearFunction_C
	lf = P.tottime * X.tottime + P.totcost * X.totcost
	lf += X("totcost*tottime") * P("fake")
	assert lf == P.tottime * X.tottime + P.totcost * X.totcost + X("totcost*tottime") * P("fake")
	# Test inplace add on attached LinearFunction_C
	m = larch.Model(utility_ca=P.tottime * X.tottime + P.totcost * X.totcost)
	m.utility_ca += X("totcost*tottime") * P("fake")
	xx = P.tottime * X.tottime + P.totcost * X.totcost + X("totcost*tottime") * P("fake")
	assert m.utility_ca == xx
